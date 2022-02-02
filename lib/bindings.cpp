#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "agent.h"

namespace py = pybind11;

int bind_update(py::array_t<float> states_vector, py::array_t<int> actions_vector, py::array_t<int> rewards_vector, py::array_t<float> next_states_vector) {
	agent_update(states_vector.data(), actions_vector.data(), rewards_vector.data(), next_states_vector.data());
	return 0;
}

int bind_act(py::array_t<float> curr_state) {
	return agent_act(curr_state.data());
}

PYBIND11_MODULE(cuda_bindings, m) {
    m.def("init", &agent_init);
	m.def("update", &bind_update);
	m.def("act", &bind_act);
	m.def("free", &agent_free);
}