/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma GCC diagnostic push
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop
#include "base_operation.h"
#include "graph_operation.h"

namespace atb_torch {
PYBIND11_MODULE(_libatb_torch, m)
{
    pybind11::class_<atb_torch::Operation>(m, "_Operation")
        .def_property("op_name", &atb_torch::Operation::GetOpName, &atb_torch::Operation::SetOpName)
        .def_property_readonly("input_names", &atb_torch::Operation::GetInputNames)
        .def_property_readonly("output_names", &atb_torch::Operation::GetOutputNames)
        .def("pre_input", &atb_torch::Operation::PreInputTensor)
        .def("pre_output", &atb_torch::Operation::PreOutputTensor)
        .def("pre_bind", &atb_torch::Operation::PreBindTensor)
        .def("set_weights", &atb_torch::Operation::SetWeights, pybind11::arg("weights") = atb_torch::TorchTensorMap())
        .def("forward", &atb_torch::Operation::Forward, pybind11::arg("input"),
             pybind11::arg("output") = atb_torch::TorchTensorMap(),
             pybind11::arg("bind") = atb_torch::TorchTensorMap());

    pybind11::class_<atb_torch::BaseOperation, atb_torch::Operation>(m, "_BaseOperation")
        .def(pybind11::init<std::string, std::string, std::string>(), pybind11::arg("op_type"),
             pybind11::arg("op_param"), pybind11::arg("op_name"))
        .def_property_readonly("op_type", &atb_torch::BaseOperation::GetOpType)
        .def_property_readonly("op_param", &atb_torch::BaseOperation::GetOpParam);

    pybind11::class_<atb_torch::GraphOperation, atb_torch::Operation>(m, "_GraphOperation")
        .def(pybind11::init<std::string>(), pybind11::arg("op_name") = "")
        .def("add_input_output", &atb_torch::GraphOperation::AddInputOutput, pybind11::arg("input"),
             pybind11::arg("output"))
        .def("add_operation", &atb_torch::GraphOperation::AddOperation, pybind11::arg("operation"),
             pybind11::arg("input"), pybind11::arg("output"))
        .def("add_reshape", &atb_torch::GraphOperation::AddReshape, pybind11::arg("input"), pybind11::arg("ouput"),
             pybind11::arg("func"))
        .def("build", &atb_torch::GraphOperation::Build)
        .def_property("execute_as_single", &atb_torch::GraphOperation::GetExecuteAsSingle,
                      &atb_torch::GraphOperation::SetExecuteAsSingle);
}
}