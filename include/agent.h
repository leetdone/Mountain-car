#pragma once

void agent_init();
int agent_update(const float* curr_state, const int* action, const int* reward, const float* next_state);
int agent_act(const float* current_state);
void agent_free();