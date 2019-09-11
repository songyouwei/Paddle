/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/imperative.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"

#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

class Layer : public imperative::Layer {
 public:
  using imperative::Layer::Layer;  // Inherit constructors

  std::vector<std::shared_ptr<imperative::VarBase>> Forward(
      const std::vector<std::shared_ptr<imperative::VarBase>> &inputs)
      override {
    PYBIND11_OVERLOAD(std::vector<std::shared_ptr<imperative::VarBase>>, Layer,
                      Forward, inputs);  // NOLINT
  }
};

// warper for pyobject to avoid imperative module depend on python
// TODO(jiabin) Add OpBase's pybind interface back to enable backward hook
class PYBIND11_HIDDEN PyCallableObject {
 public:
  PyCallableObject(std::shared_ptr<py::object> py_obj_ptr)
      : py_obj_ptr_(std::move(py_obj_ptr)) {}
  ~PyCallableObject() {
    py::call_guard<py::gil_scoped_acquire>();
    py_obj_ptr_.reset();
  }
  void operator()() {
    py::call_guard<py::gil_scoped_acquire>();
    py_obj_ptr_->operator()(this);
  }

 private:
  std::shared_ptr<py::object> py_obj_ptr_;
};

// Function like obj.attr_name in Python.
static PyObject *GetPythonAttribute(PyObject *obj, const char *attr_name) {
  // NOTE(zjl): PyObject_GetAttrString would return nullptr when attr_name
  // is not inside obj, but it would also set the error flag of Python.
  // If the error flag is set in C++, C++ code would not raise Exception,
  // but Python would raise Exception once C++ call ends.
  // To avoid unexpected Exception raised in Python, we check whether
  // attribute exists before calling PyObject_GetAttrString.
  //
  // Caution: PyObject_GetAttrString would increase reference count of PyObject.
  // Developer should call Py_DECREF manually after the attribute is not used.
  if (PyObject_HasAttrString(obj, attr_name)) {
    return PyObject_GetAttrString(obj, attr_name);
  } else {
    return nullptr;
  }
}

template <typename T>
static T PyObjectCast(PyObject *obj) {
  try {
    return py::cast<T>(py::handle(obj));
  } catch (py::cast_error &) {
    PADDLE_THROW("Python object is not type of %s", typeid(T).name());
  }
}

// NOTE(zjl): py::handle is a very light wrapper of PyObject *.
// Unlike py::object, py::handle does not change reference count of PyObject *.
static std::vector<std::shared_ptr<imperative::VarBase>>
GetVarBaseListFromPyHandle(const py::handle &handle) {
  PyObject *py_obj = handle.ptr();  // get underlying PyObject
  // Python None is not nullptr in C++!
  if (!py_obj || py_obj == Py_None) {
    return {};
  }

  const char *kIVarField = "_ivar";
  //  PyObject *py_ivar = GetPythonAttribute(py_obj, kIVarField);
  std::vector<std::shared_ptr<imperative::VarBase>> result;

  if (PyList_Check(py_obj)) {  // List of Variable
    size_t len = PyList_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar = PyObject_GetAttrString(PyList_GET_ITEM(py_obj, i),
                                                 kIVarField);  // will incref
      if (!py_ivar) {
        VLOG(3) << "List of VarBase";
        py_ivar = PyList_GET_ITEM(py_obj, i);  // is VarBase
        Py_INCREF(py_ivar);                    // new ref instead
      }
      PADDLE_ENFORCE_NOT_NULL(py_ivar);
      auto ivar = PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar);
      result.emplace_back(ivar);
      VLOG(3) << "input " << i << "th VarBase: " << ivar->Name();
      Py_DECREF(py_ivar);
    }
  } else if (PyTuple_Check(py_obj)) {  // Tuple of Variable
    size_t len = PyTuple_GET_SIZE(py_obj);
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      PyObject *py_ivar =
          PyObject_GetAttrString(PyTuple_GET_ITEM(py_obj, i), kIVarField);
      if (!py_ivar) {
        VLOG(3) << "Tuple of VarBase";
        py_ivar = PyTuple_GET_ITEM(py_obj, i);  // is VarBase
        Py_INCREF(py_ivar);                     // new ref instead
      }
      PADDLE_ENFORCE_NOT_NULL(py_ivar);
      auto ivar = PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar);
      result.emplace_back(ivar);
      VLOG(3) << "input " << i << "th VarBase: " << ivar->Name();
      Py_DECREF(py_ivar);
    }
  } else {  // Variable
    PyObject *py_ivar = GetPythonAttribute(py_obj, kIVarField);
    if (!py_ivar) {  // is VarBase
      py_ivar = py_obj;
      Py_INCREF(py_ivar);  // new ref instead
      VLOG(3) << "single VarBase";
    } else {
      VLOG(3) << "single Variable";
    }
    auto ivar = PyObjectCast<std::shared_ptr<imperative::VarBase>>(py_ivar);
    result.emplace_back(ivar);
    VLOG(3) << "input VarBase: " << ivar->Name();
    Py_DECREF(py_ivar);
  }
  //  else {
  //    PADDLE_THROW(
  //        "unsupported type %s, must be Variable, list[Variable] or "
  //        "tuple[Variable]",
  //        py::str(handle));
  //  }

  return result;
}

using PyNameVarBaseMap = std::unordered_map<std::string, py::handle>;
using PyNameAttributeMap = std::unordered_map<std::string, py::handle>;

static imperative::NameVarBaseMap ConvertToNameVarBaseMap(
    const PyNameVarBaseMap &map) {
  imperative::NameVarBaseMap result;
  for (auto &pair : map) {
    auto var_vec = GetVarBaseListFromPyHandle(pair.second);
    if (!var_vec.empty()) {
      result.emplace(pair.first, std::move(var_vec));
    }
  }

  PADDLE_ENFORCE_EQ(PyErr_Occurred() == nullptr, true,
                    py::str(py::handle(PyErr_Occurred())));
  return result;
}

static framework::AttributeMap ConvertToAttributeMap(
    const PyNameAttributeMap &map) {
  framework::AttributeMap result;
  for (auto &pair : map) {
    auto it = PyObjectCast<framework::Attribute>(pair.second.ptr());
    result[pair.first] = it;
  }
  return result;
}

static std::string GetTypeName(const imperative::VarBase &var) {
  if (var.Type() == framework::proto::VarType::RAW) {
    return "RAW";
  } else if (!var.Var().IsInitialized()) {
    return "nullptr";
  } else {
    return framework::ToTypeName(var.Var().Type());
  }
}

inline std::string Utils_unpackString(PyObject *obj) {
  if (PyBytes_Check(obj)) {
    size_t size = PyBytes_GET_SIZE(obj);
    return std::string(PyBytes_AS_STRING(obj), size);
  }
  if (PyUnicode_Check(obj)) {
#if PY_MAJOR_VERSION == 2
    PyObject *s = PyUnicode_AsUTF8String(obj);
    if (!s) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    size_t size = PyBytes_GET_SIZE(s);
    return std::string(PyBytes_AS_STRING(s), size);
#else
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    return std::string(data, (size_t)size);
#endif
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object");
}

template <typename T>
static T ConvertPyObjectToMap(PyObject *obj) {
  T result;
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    result[Utils_unpackString(key)] = value;
  }
  return result;
}

// _C.Tracer
struct PyTracer {
  PyObject_HEAD imperative::Tracer tracer;
};
static PyObject *PyTracer_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwargs) {
  PyObject *obj = type->tp_alloc(type, 0);
  return obj;
}
static int PyTracer_init(PyTracer *self, PyObject *args, PyObject *kwargs) {
  new (&self->tracer) imperative::Tracer();
  return 0;
}

// args format:
//  type, inputs_size, outputs_size, attrs_size,
//  input_1_key, input_1_value, ...,
//  output_1_key, output_1_num, ...,
//  attr_1_key, attr_1_value, ...,
//  place, stop_gradient
PyObject *PyTracer_trace_tuple_return_out(PyTracer *self, PyObject *args) {
  VLOG(3) << "PyTracer_trace_tuple_return_out";
  Py_ssize_t args_size = PyTuple_GET_SIZE(args);
  Py_ssize_t idx = 0;
  std::string type = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
  Py_ssize_t inputs_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
  Py_ssize_t outputs_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
  Py_ssize_t attrs_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
  PADDLE_ENFORCE_EQ(args_size,
                    6 + (inputs_size + outputs_size + attrs_size) * 2);

  VLOG(3) << "type:" << type;
  VLOG(3) << "inputs_size:" << inputs_size;
  VLOG(3) << "outputs_size:" << outputs_size;
  VLOG(3) << "attrs_size:" << attrs_size;

  imperative::NameVarBaseMap inputs, outputs;
  framework::AttributeMap attrs;

  for (Py_ssize_t i = 0; i < inputs_size; ++i) {
    std::string key = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
    auto value =
        GetVarBaseListFromPyHandle(py::handle(PyTuple_GET_ITEM(args, idx++)));
    if (!value.empty()) {
      inputs.emplace(key, std::move(value));
    }
  }
  for (Py_ssize_t i = 0; i < outputs_size; ++i) {
    std::string key = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
    Py_ssize_t num = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
    std::vector<std::shared_ptr<imperative::VarBase>> value;
    value.reserve(num);
    VLOG(3) << "num of new outs: " << num;
    for (Py_ssize_t j = 0; j < num; ++j) {
      auto var_name = type + "." + key + std::to_string(j);
      VLOG(3) << "new out var: " << var_name;
      std::shared_ptr<imperative::VarBase> var(
          new imperative::VarBase(var_name));
      auto *tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();
      tensor->Resize(framework::make_ddim({}));
      value.emplace_back(var);
    }
    outputs.emplace(key, value);
  }
  for (Py_ssize_t i = 0; i < attrs_size; ++i) {
    std::string key = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
    auto value =
        PyObjectCast<framework::Attribute>(PyTuple_GET_ITEM(args, idx++));
    if (!value.empty()) {
      attrs.emplace(key, std::move(value));
    }
  }

  auto place = PyTuple_GET_ITEM(args, idx++);
  auto stop_gradient = PyObject_IsTrue(PyTuple_GET_ITEM(args, idx));

  auto place_class_name = Utils_unpackString(
      GetPythonAttribute(GetPythonAttribute(place, "__class__"), "__name__"));
  if (place_class_name == "CPUPlace") {
    py::gil_scoped_release release;
    self->tracer.TraceOp(type, std::move(inputs), outputs, std::move(attrs),
                         PyObjectCast<platform::CPUPlace>(place),
                         stop_gradient);
  } else {
    py::gil_scoped_release release;
    self->tracer.TraceOp(type, std::move(inputs), outputs, std::move(attrs),
                         PyObjectCast<platform::CUDAPlace>(place),
                         stop_gradient);
  }
  VLOG(3) << "ready to return outs";
  VLOG(3) << "size of outs: " << outputs.size();
  //             for (Py_ssize_t i = 0; i < outputs_size; ++i) {
  //               PADDLE_ENFORCE_NOT_NULL(outputs["Out"][i].get());
  //             }
  py::object outobj =
      py::cast(outputs["Out"][0], py::return_value_policy::automatic);
  py::handle out = outobj.release();  // release is required
  VLOG(3) << "out refcnt: " << out.ref_count();
  return out.ptr();
}

// args format:
//  type, input_size, output_size, attrs_size,
//  input_1_key, input_1_value, ...,
//  output_1_key, output_1_value, ...,
//  attr_1_key, attr_1_value, ...,
//  place, stop_gradient
PyObject *PyTracer_trace_tuple(PyTracer *self, PyObject *args) {
  Py_ssize_t args_size = PyTuple_GET_SIZE(args);
  Py_ssize_t idx = 0;
  std::string type = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
  Py_ssize_t inputs_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
  Py_ssize_t outputs_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
  Py_ssize_t attrs_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
  PADDLE_ENFORCE_EQ(args_size,
                    6 + (inputs_size + outputs_size + attrs_size) * 2);

  imperative::NameVarBaseMap inputs, outputs;
  framework::AttributeMap attrs;

  for (Py_ssize_t i = 0; i < inputs_size; ++i) {
    std::string key = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
    auto value =
        GetVarBaseListFromPyHandle(py::handle(PyTuple_GET_ITEM(args, idx++)));
    if (!value.empty()) {
      inputs.emplace(key, std::move(value));
    }
  }
  for (Py_ssize_t i = 0; i < outputs_size; ++i) {
    std::string key = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
    auto value =
        GetVarBaseListFromPyHandle(py::handle(PyTuple_GET_ITEM(args, idx++)));
    if (!value.empty()) {
      outputs.emplace(key, std::move(value));
    }
  }
  for (Py_ssize_t i = 0; i < attrs_size; ++i) {
    std::string key = Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
    auto value =
        PyObjectCast<framework::Attribute>(PyTuple_GET_ITEM(args, idx++));
    if (!value.empty()) {
      attrs.emplace(key, std::move(value));
    }
  }

  auto place = PyTuple_GET_ITEM(args, idx++);
  auto stop_gradient = PyObject_IsTrue(PyTuple_GET_ITEM(args, idx));

  auto place_class_name = Utils_unpackString(
      GetPythonAttribute(GetPythonAttribute(place, "__class__"), "__name__"));
  if (place_class_name == "CPUPlace") {
    py::gil_scoped_release release;
    self->tracer.TraceOp(
        type, std::move(inputs), std::move(outputs), std::move(attrs),
        PyObjectCast<platform::CPUPlace>(place), stop_gradient);
  } else {
    py::gil_scoped_release release;
    self->tracer.TraceOp(
        type, std::move(inputs), std::move(outputs), std::move(attrs),
        PyObjectCast<platform::CUDAPlace>(place), stop_gradient);
  }
  Py_RETURN_NONE;
}

PyObject *PyTracer_trace(PyTracer *self, PyObject *args, PyObject *kwargs) {
  const char *type = nullptr;
  PyObject *inputs = nullptr;
  PyObject *outputs = nullptr;
  PyObject *attrs = nullptr;
  PyObject *place = nullptr;
  unsigned char stop_gradient = 0;

  const char *accepted_kwargs[] = {"type",  "inputs",        "outputs", "attrs",
                                   "place", "stop_gradient", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "sOOOO|b", const_cast<char **>(accepted_kwargs), &type,
          &inputs, &outputs, &attrs, &place, &stop_gradient))
    Py_RETURN_NONE;

  auto ins_map =
      ConvertToNameVarBaseMap(ConvertPyObjectToMap<PyNameVarBaseMap>(inputs));
  auto outs_map =
      ConvertToNameVarBaseMap(ConvertPyObjectToMap<PyNameVarBaseMap>(outputs));
  auto attrs_map =
      ConvertToAttributeMap(ConvertPyObjectToMap<PyNameAttributeMap>(attrs));

  auto place_class_name = Utils_unpackString(
      GetPythonAttribute(GetPythonAttribute(place, "__class__"), "__name__"));
  if (place_class_name == "CPUPlace") {
    py::gil_scoped_release release;
    self->tracer.TraceOp(
        type, std::move(ins_map), std::move(outs_map), std::move(attrs_map),
        PyObjectCast<platform::CPUPlace>(place), stop_gradient);
  } else {
    py::gil_scoped_release release;
    self->tracer.TraceOp(
        type, std::move(ins_map), std::move(outs_map), std::move(attrs_map),
        PyObjectCast<platform::CUDAPlace>(place), stop_gradient);
  }
  Py_RETURN_NONE;
}
static struct PyMethodDef PyTracer_methods[] = {
    {const_cast<char *>("trace"), (PyCFunction)PyTracer_trace,
     METH_VARARGS | METH_KEYWORDS, nullptr},
    {const_cast<char *>("trace_tuple"), (PyCFunction)PyTracer_trace_tuple,
     METH_VARARGS | METH_KEYWORDS, nullptr},
    {const_cast<char *>("trace_tuple_return_out"),
     (PyCFunction)PyTracer_trace_tuple_return_out, METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {nullptr, nullptr, METH_NOARGS, nullptr}};
static void PyTracer_dealloc(PyTracer *self) {
  self->tracer.~Tracer();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}
PyTypeObject PyTracerType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "_C.Tracer", /* tp_name */
    sizeof(PyTracer),                              /* tp_basicsize */
    0,                                             /* tp_itemsize */
    (destructor)PyTracer_dealloc,                  /* tp_dealloc */
    nullptr,                                       /* tp_print */
    nullptr,                                       /* tp_getattr */
    nullptr,                                       /* tp_setattr */
    nullptr,                                       /* tp_reserved */
    nullptr,                                       /* tp_repr */
    nullptr,                                       /* tp_as_number */
    nullptr,                                       /* tp_as_sequence */
    nullptr,                                       /* tp_as_mapping */
    nullptr,                                       /* tp_hash  */
    nullptr,                                       /* tp_call */
    nullptr,                                       /* tp_str */
    nullptr,                                       /* tp_getattro */
    nullptr,                                       /* tp_setattro */
    nullptr,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,      /* tp_flags */
    "PyTracer",                                    /* tp_doc */
    nullptr,                                       /* tp_traverse */
    nullptr,                                       /* tp_clear */
    nullptr,                                       /* tp_richcompare */
    0,                                             /* tp_weaklistoffset */
    nullptr,                                       /* tp_iter */
    nullptr,                                       /* tp_iternext */
    PyTracer_methods,                              /* tp_methods */
    nullptr,                                       /* tp_members */
    nullptr,                                       /* tp_getset */
    nullptr,                                       /* tp_base */
    nullptr,                                       /* tp_dict */
    nullptr,                                       /* tp_descr_get */
    nullptr,                                       /* tp_descr_set */
    0,                                             /* tp_dictoffset */
    reinterpret_cast<initproc>(PyTracer_init),     /* tp_init */
    nullptr,                                       /* tp_alloc */
    PyTracer_new,                                  /* tp_new */
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    0,
#if PY_MAJOR_VERSION >= 3
    nullptr,
#endif
};

bool PyTracer_initModule(PyObject *module) {
  if (PyType_Ready(&PyTracerType) < 0) return false;
  Py_INCREF(&PyTracerType);
  PyModule_AddObject(module, "Tracer",
                     reinterpret_cast<PyObject *>(&PyTracerType));
  return true;
}

PyObject *initPythonCModule() {
  PyObject *module;
#define ASSERT_TRUE(cmd) \
  if (!(cmd)) return nullptr
#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("_C", {}));
#else
  static struct PyModuleDef m = {PyModuleDef_HEAD_INIT,
                                 "_C",
                                 nullptr,
                                 -1,
                                 {},
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr};
  ASSERT_TRUE(module = PyModule_Create(&m));
#endif
  ASSERT_TRUE(PyTracer_initModule(module));
  return module;
}

void BindPythonCModule(py::module *m_ptr) {
  VLOG(3) << "BindPythonCModule";
  auto &m = *m_ptr;
  PyObject *pythonc_module = initPythonCModule();
  Py_INCREF(pythonc_module);
  PyModule_AddObject(m.ptr(), (const char *)"_C", pythonc_module);
}

// Bind Methods
void BindImperative(py::module *m_ptr) {
  auto &m = *m_ptr;

  py::class_<imperative::detail::BackwardStrategy> backward_strategy(
      m, "BackwardStrategy", R"DOC(

    BackwardStrategy is a descriptor of a how to run the backward process. Now it has:

    1. :code:`sort_sum_gradient`, which will sum the gradient by the reverse order of trace.

    Examples:

        .. code-block:: python

          import numpy as np
          import paddle.fluid as fluid
          from paddle.fluid import FC

          x = np.ones([2, 2], np.float32)
          with fluid.dygraph.guard():
              inputs2 = []
              for _ in range(10):
                  inputs2.append(fluid.dygraph.base.to_variable(x))
              ret2 = fluid.layers.sums(inputs2)
              loss2 = fluid.layers.reduce_sum(ret2)
              backward_strategy = fluid.dygraph.BackwardStrategy()
              backward_strategy.sort_sum_gradient = True
              loss2.backward(backward_strategy)
      )DOC");
  backward_strategy.def(py::init())
      .def_property("sort_sum_gradient",
                    [](const imperative::detail::BackwardStrategy &self) {
                      return self.sorted_sum_gradient_;
                    },
                    [](imperative::detail::BackwardStrategy &self,
                       bool sorted_sum_gradient) {
                      self.sorted_sum_gradient_ = sorted_sum_gradient;
                    });

  m.def("start_imperative_gperf_profiler",
        []() { imperative::StartProfile(); });

  m.def("stop_imperative_gperf_profiler", []() { imperative::StopProfile(); });

  m.def("_is_dygraph_debug_enabled",
        []() { return imperative::IsDebugEnabled(); });
  m.def("_dygraph_debug_level", []() { return imperative::GetDebugLevel(); });

  py::class_<imperative::VarBase, std::shared_ptr<imperative::VarBase>>(
      m, "VarBase",
      R"DOC()DOC")
      .def_static("_alive_vars", &imperative::VarBase::AliveVarNames)
      .def("__init__",
           [](imperative::VarBase &self, const std::string &name,
              framework::proto::VarType::Type type,
              framework::proto::VarType::Type dtype,
              const std::vector<int> &dims, bool stop_gradient,
              bool persistable) {
             new (&self) imperative::VarBase(name);
             self.SetPersistable(persistable);
             self.SetType(type);
             self.SetDataType(dtype);
             self.SetStopGradient(stop_gradient);
             if (type == framework::proto::VarType::LOD_TENSOR) {
               auto *tensor =
                   self.MutableVar()->GetMutable<framework::LoDTensor>();
               tensor->Resize(framework::make_ddim(dims));
             }
           })
      .def("_run_backward",
           [](imperative::VarBase &self,
              const imperative::detail::BackwardStrategy &bckst,
              const imperative::Tracer &tracer) {
             // TODO(jiabin): when we impl more backward execution we can select
             // them

             imperative::Engine *engine = tracer.GetDefaultEngine();
             VLOG(3) << "Start backward";
             engine->Init(&self, bckst);
             engine->Execute();
             VLOG(3) << "Finish backward";
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_run_backward",
           [](imperative::VarBase &self,
              const imperative::detail::BackwardStrategy &bckst,
              const py::handle &pytracer) {
             // TODO(jiabin): when we impl more backward execution we can select
             // them

             auto _pytracer = reinterpret_cast<PyTracer *>(pytracer.ptr());
             imperative::Engine *engine = _pytracer->tracer.GetDefaultEngine();
             VLOG(3) << "Start backward";
             engine->Init(&self, bckst);
             engine->Execute();
             VLOG(3) << "Finish backward";
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_grad_name", &imperative::VarBase::GradVarName)
      .def("_grad_value",
           [](imperative::VarBase &self) {
             return self.MutableGradVar()->Get<framework::LoDTensor>();
           },
           py::return_value_policy::reference)
      .def("_clear_gradient", &imperative::VarBase::ClearGradient)
      .def("_grad_ivar",
           [](const imperative::VarBase &self) {
             auto &grad_var = self.GradVarBase();
             if (grad_var && grad_var->Var().IsInitialized()) {
               return grad_var;
             } else {
               return std::shared_ptr<imperative::VarBase>(nullptr);
             }
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CPUPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CUDAPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("value", [](imperative::VarBase &self) { return self.MutableVar(); },
           py::return_value_policy::reference)
      .def_property("name", &imperative::VarBase::Name,
                    &imperative::VarBase::SetName)
      .def_property_readonly(
          "shape",
          [](imperative::VarBase &self) {
            if (self.Var().IsType<framework::LoDTensor>()) {
              return framework::vectorize<int>(
                  self.Var().Get<framework::LoDTensor>().dims());
            } else {
              VLOG(2) << "It is meaningless to get shape of variable type "
                      << GetTypeName(self);
              return std::vector<int>();
            }
          })
      .def_property_readonly("type", &imperative::VarBase::Type)
      .def_property_readonly("dtype", &imperative::VarBase::DataType)
      .def_property("persistable", &imperative::VarBase::Persistable,
                    &imperative::VarBase::SetPersistable)
      .def_property("stop_gradient", &imperative::VarBase::StopGradient,
                    &imperative::VarBase::SetStopGradient);

  py::class_<imperative::Layer, Layer /* <--- trampoline*/> layer(m, "Layer");
  layer.def(py::init<>())
      .def("forward",
           [](imperative::Layer &self,
              const std::vector<std::shared_ptr<imperative::VarBase>> &inputs) {
             return self.Forward(inputs);
           });

  py::class_<imperative::Tracer>(m, "Tracer", "")
      .def("__init__",
           [](imperative::Tracer &self) { new (&self) imperative::Tracer(); })
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CUDAPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
             }
           })
      //      .def("addone", [](imperative::Tracer &self, int a) {
      //              return a+1;
      //      })
      //      .def("do_dict",[](imperative::Tracer &self, py::handle &vv0,
      //      py::handle &vv1, py::handle &vv2) {
      //              long result = 0;
      //              for (auto vv : {vv0, vv1, vv2}) {
      //
      ////                for (long i = 0; i < PyList_Size(vv.ptr()); ++i) {
      ////                  PyObject *py_vec = PyList_GET_ITEM(vv.ptr(), i);
      ////                  for (long j = 0; j < PyList_Size(py_vec); ++j) {
      ////                    PyLong_AsLong(PyList_GET_ITEM(py_vec, j));
      ////                  }
      ////                }
      //                  result += PyLong_AsLong(
      //                  PyList_GET_ITEM(PyList_GET_ITEM(vv.ptr(), 0), 0));
      //              }
      //              return result;
      //      })
      //      .def("trace_unpacked",
      //              [](imperative::Tracer &self,
      //                      const std::string &type,
      //                      const imperative::VarBase &X,
      //                      const imperative::VarBase &Y,
      //                      const imperative::VarBase &Out,
      //                      const int x_num_col_dims,
      //                      const int y_num_col_dims,
      //                      const platform::CPUPlace &place,
      //                      bool trace_backward) {
      //                  return 0;
      //      })
      .def("trace_tuple_return_out",
           [](imperative::Tracer &self, py::handle _args) {
             VLOG(3) << "trace_tuple_return_out";
             PyObject *args = _args.ptr();
             Py_ssize_t args_size = PyTuple_GET_SIZE(args);
             Py_ssize_t idx = 0;
             std::string type =
                 Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
             Py_ssize_t inputs_size =
                 PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
             Py_ssize_t outputs_size =
                 PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
             Py_ssize_t attrs_size =
                 PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
             PADDLE_ENFORCE_EQ(
                 args_size, 6 + (inputs_size + outputs_size + attrs_size) * 2);

             VLOG(3) << "type:" << type;
             VLOG(3) << "inputs_size:" << inputs_size;
             VLOG(3) << "outputs_size:" << outputs_size;
             VLOG(3) << "attrs_size:" << attrs_size;

             imperative::NameVarBaseMap inputs, outputs;
             framework::AttributeMap attrs;

             for (Py_ssize_t i = 0; i < inputs_size; ++i) {
               std::string key =
                   Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
               auto value = GetVarBaseListFromPyHandle(
                   py::handle(PyTuple_GET_ITEM(args, idx++)));
               if (!value.empty()) {
                 inputs.emplace(key, std::move(value));
               }
             }
             for (Py_ssize_t i = 0; i < outputs_size; ++i) {
               std::string key =
                   Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
               Py_ssize_t num = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
               std::vector<std::shared_ptr<imperative::VarBase>> value;
               value.reserve(num);
               VLOG(3) << "num of new outs: " << num;
               for (Py_ssize_t j = 0; j < num; ++j) {
                 auto var_name = type + "." + key + std::to_string(j);
                 VLOG(3) << "new out var: " << var_name;
                 std::shared_ptr<imperative::VarBase> var(
                     new imperative::VarBase(var_name));
                 auto *tensor =
                     var->MutableVar()->GetMutable<framework::LoDTensor>();
                 tensor->Resize(framework::make_ddim({}));
                 value.emplace_back(var);
               }
               outputs.emplace(key, value);
             }
             for (Py_ssize_t i = 0; i < attrs_size; ++i) {
               std::string key =
                   Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
               auto value = PyObjectCast<framework::Attribute>(
                   PyTuple_GET_ITEM(args, idx++));
               if (!value.empty()) {
                 attrs.emplace(key, std::move(value));
               }
             }

             auto place = PyTuple_GET_ITEM(args, idx++);
             auto stop_gradient = PyObject_IsTrue(PyTuple_GET_ITEM(args, idx));

             auto place_class_name = Utils_unpackString(GetPythonAttribute(
                 GetPythonAttribute(place, "__class__"), "__name__"));
             if (place_class_name == "CPUPlace") {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(inputs), outputs, std::move(attrs),
                            PyObjectCast<platform::CPUPlace>(place),
                            stop_gradient);
             } else {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(inputs), outputs, std::move(attrs),
                            PyObjectCast<platform::CUDAPlace>(place),
                            stop_gradient);
             }
             VLOG(3) << "ready to return outs";
             VLOG(3) << "size of outs: " << outputs.size();
             //             for (Py_ssize_t i = 0; i < outputs_size; ++i) {
             //               PADDLE_ENFORCE_NOT_NULL(outputs["Out"][i].get());
             //             }
             return outputs["Out"][0];
           })
      .def("trace_tuple",
           [](imperative::Tracer &self, py::handle _args) {
             PyObject *args = _args.ptr();
             Py_ssize_t args_size = PyTuple_GET_SIZE(args);
             Py_ssize_t idx = 0;
             std::string type =
                 Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
             Py_ssize_t inputs_size =
                 PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
             Py_ssize_t outputs_size =
                 PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
             Py_ssize_t attrs_size =
                 PyLong_AsSsize_t(PyTuple_GET_ITEM(args, idx++));
             PADDLE_ENFORCE_EQ(
                 args_size, 6 + (inputs_size + outputs_size + attrs_size) * 2);

             imperative::NameVarBaseMap inputs, outputs;
             framework::AttributeMap attrs;

             for (Py_ssize_t i = 0; i < inputs_size; ++i) {
               std::string key =
                   Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
               auto value = GetVarBaseListFromPyHandle(
                   py::handle(PyTuple_GET_ITEM(args, idx++)));
               if (!value.empty()) {
                 inputs.emplace(key, std::move(value));
               }
             }
             for (Py_ssize_t i = 0; i < outputs_size; ++i) {
               std::string key =
                   Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
               auto value = GetVarBaseListFromPyHandle(
                   py::handle(PyTuple_GET_ITEM(args, idx++)));
               if (!value.empty()) {
                 outputs.emplace(key, std::move(value));
               }
             }
             for (Py_ssize_t i = 0; i < attrs_size; ++i) {
               std::string key =
                   Utils_unpackString(PyTuple_GET_ITEM(args, idx++));
               auto value = PyObjectCast<framework::Attribute>(
                   PyTuple_GET_ITEM(args, idx++));
               if (!value.empty()) {
                 attrs.emplace(key, std::move(value));
               }
             }

             auto place = PyTuple_GET_ITEM(args, idx++);
             auto stop_gradient = PyObject_IsTrue(PyTuple_GET_ITEM(args, idx));

             auto place_class_name = Utils_unpackString(GetPythonAttribute(
                 GetPythonAttribute(place, "__class__"), "__name__"));
             if (place_class_name == "CPUPlace") {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(inputs), outputs, std::move(attrs),
                            PyObjectCast<platform::CPUPlace>(place),
                            stop_gradient);
             } else {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(inputs), outputs, std::move(attrs),
                            PyObjectCast<platform::CUDAPlace>(place),
                            stop_gradient);
             }

             return outputs;
           })
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const PyNameVarBaseMap &ins, const PyNameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CPUPlace &place,
              bool trace_backward) {
             auto ins_map = ConvertToNameVarBaseMap(ins);
             auto outs_map = ConvertToNameVarBaseMap(outs);
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins_map), std::move(outs_map),
                            std::move(attrs), place, trace_backward);
             }
           });

  // define parallel context
  py::class_<imperative::ParallelStrategy> parallel_strategy(
      m, "ParallelStrategy", "");
  parallel_strategy.def(py::init())
      .def_property(
          "nranks",
          [](const imperative::ParallelStrategy &self) { return self.nranks_; },
          [](imperative::ParallelStrategy &self, int nranks) {
            self.nranks_ = nranks;
          })
      .def_property("local_rank",
                    [](const imperative::ParallelStrategy &self) {
                      return self.local_rank_;
                    },
                    [](imperative::ParallelStrategy &self, int local_rank) {
                      self.local_rank_ = local_rank;
                    })
      .def_property(
          "trainer_endpoints",
          [](const imperative::ParallelStrategy &self) {
            return self.trainer_endpoints_;
          },
          [](imperative::ParallelStrategy &self, std::vector<std::string> eps) {
            self.trainer_endpoints_ = eps;
          })
      .def_property("current_endpoint",
                    [](const imperative::ParallelStrategy &self) {
                      return self.current_endpoint_;
                    },
                    [](imperative::ParallelStrategy &self,
                       const std::string &ep) { self.current_endpoint_ = ep; });
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  py::class_<imperative::NCCLParallelContext> nccl_ctx(m,
                                                       "NCCLParallelContext");

  nccl_ctx
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CUDAPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); });
#endif
}

}  // namespace pybind
}  // namespace paddle
