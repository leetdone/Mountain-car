#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <time.h>

const unsigned int PARALLEL = 10;
const unsigned int SEGMENTS = 100;

// Average w and theta weights
static float* w = NULL;
static float* theta = NULL;

// Device vars for cuda_act
static int* d_action = 0;
static float* curr_state_act = NULL;
static curandState* rand_state = NULL;

// Device vars for cuda_update
static float* states_vector = NULL;
static int* actions_vector = NULL;
static int* rewards_vector = NULL;
static float* next_states_vector = NULL;
static int* average_reward = NULL;

// Estimate the value function using a simple linear model
__device__ float approximate_value(float* discrete_state, float* w) { // good
	float result = 0;

	for (int i = 0; i < 2 * SEGMENTS; i++) {
		result += discrete_state[i] * w[i];
	}

	return result;
}

// out_preference should be a float array with length 3
// out_preference_grad should be a float array with length 600*3 = 1800
__device__ void calc_preference(float* discrete_state, float *theta, float* out_preference, float* out_preference_grad) { // good
	for (int i = 0; i < 3; i++) {
		out_preference[i] = 0;
	}

	for (int a = 0; a < 3; a++)
	{
		float action_vector[3] = { 0 };
		action_vector[a] = 1;
		float s_a[600] = { 0 };

		int index = 0;
		for (int s_i = 0; s_i < 200; s_i++)//discrete_state has 200 floats, action_vector has 3 floats
		{
			for (int a_i = 0; a_i < 3; a_i++)
			{
				s_a[index] = discrete_state[s_i] * action_vector[a_i];
				index++;
			}

		}

		//dot(s_a, self. theta), s_a has 600 floats, theta has 600 floats
		float dot_product = 0;
		for (int i = 0; i < 600; i++)
		{
			dot_product += s_a[i] * theta[i];

			out_preference_grad[(600 * a) + i] = s_a[i];
		}

		out_preference[a] = dot_product;
	}
}

__device__ void calc_policy(float* discrete_state, float* theta, float* out_policy) { // good
	float policy[3];
	float preference_grad[1800];
	calc_preference(discrete_state, theta, policy, preference_grad);

	float max_policy = policy[0];//find the maximum policy
	for (int i = 1; i < 3; i++)
	{
		if (policy[i] > max_policy)
		{
			max_policy = policy[i];
		}
	}

	float numerator[3] = { 0 };
	for (int i = 0; i < 3; i++)
	{
		numerator[i] = expf(policy[i] - max_policy);
	}

	float denominator = 0;
	for (int i = 0; i < 3; i++) {
		denominator += numerator[i];
	}

	for (int i = 0; i < 3; i++) {
		out_policy[i] = numerator[i] / denominator;
	}
}

__device__ void calc_policy_gradient(int action, float* discrete_state, float* theta, float* out_policy_grad) { // good
	float policy[3];
	float preference_grad[1800];
	calc_preference(discrete_state, theta, policy, preference_grad);

	float h_s_a_w[3][600];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2 * 3 * SEGMENTS; j++) {
			h_s_a_w[i][j] = preference_grad[i * 2 * 3 * SEGMENTS + j] * theta[j];
		}
	}

	float f[600];
	float f_prime[600];
	for (int i = 0; i < 2 * 3 * SEGMENTS; i++) {
		f[i] = expf(h_s_a_w[action][i]);
		f_prime[i] = f[i] * preference_grad[action * 2 * 3 * SEGMENTS + i];
	}

	float g[600];
	float g_prime[600];
	for (int i = 0; i < 600; i++) {
		g[i] = expf(h_s_a_w[0][i]) + expf(h_s_a_w[1][i]) + expf(h_s_a_w[2][i]);
		g_prime[i] = expf(h_s_a_w[0][i]) * preference_grad[i] + expf(h_s_a_w[1][i]) * preference_grad[600 + i] + expf(h_s_a_w[2][i]) * preference_grad[1200 + i];
	}

	for (int i = 0; i < 600; i++) {
		float numerator = f_prime[i] * g[i] - f[i] * g_prime[i];
		float denominator = powf(g[i], 2);
		out_policy_grad[i] = numerator / denominator;
	}
}

__device__ void segment_state(float* curr_state, float* discrete_state) { // good
	double pos_segment = (0.6 + 1.2) / SEGMENTS;
	double vel_segment = (0.7 + 0.7) / SEGMENTS;

	for (int i = 0; i < SEGMENTS * 2; i++)
	{
		discrete_state[i] = 0;
	}

	int coarse_pos_index = int((curr_state[0] + 1.2) / pos_segment);
	int coarse_vel_index = int((curr_state[1] + 0.7) / vel_segment);
	discrete_state[coarse_pos_index] = 1;
	discrete_state[coarse_vel_index + SEGMENTS] = 1;
}

__global__ void cuda_agent_init(curandState* rand_state) {
	// We only need 1 random state because agent actions are serial and not parallel.
	// This is because the environment runs in Python so only one agent can act at a time.
	curand_init(123, 0, 0, rand_state);
}

__global__ void cuda_update(float* states_vector, int* actions_vector, int* rewards_vector, float* next_states_vector, float* w, float* theta, int* average_reward) {
	__shared__ float thread_w[2 * SEGMENTS * PARALLEL];
	__shared__ float thread_theta[2 * SEGMENTS * 3 * PARALLEL];
	__shared__ int thread_rewards[PARALLEL];

	for (int i = 0; i < 2 * SEGMENTS; i++) {
		thread_w[2 * SEGMENTS * threadIdx.x + i] = w[i];
	}

	for (int i = 0; i < 2 * SEGMENTS * 3; i++) {
		thread_theta[2 * SEGMENTS * 3 * threadIdx.x + i] = theta[i];
	}

	float* states = &states_vector[threadIdx.x * 200 * 2];
	int* actions = &actions_vector[threadIdx.x * 200];
	int* rewards = &rewards_vector[threadIdx.x * 200];
	float* next_states = &next_states_vector[threadIdx.x * 200 * 2];

	int ep_reward = 0;

	for (int j = 0; j < 200; j++) {
		float* curr_state = &states[j * 2];
		int action = actions[j];
		int reward = rewards[j];
		float* next_state = &next_states[j * 2];

		float curr_discrete[SEGMENTS * 2] = { 0 };
		float next_discrete[SEGMENTS * 2] = { 0 };
		segment_state(curr_state, curr_discrete);
		segment_state(next_state, next_discrete);

		float approx_next_val = approximate_value(next_discrete, &thread_w[threadIdx.x * 2 * SEGMENTS]);
		float approx_curr_val = approximate_value(curr_discrete, &thread_w[threadIdx.x * 2 * SEGMENTS]);

		float delta_t = reward + approx_next_val - approx_curr_val;

		for (int i = 0; i < 2 * SEGMENTS; i++) {
			thread_w[threadIdx.x * 2 * SEGMENTS + i] += 0.01f * delta_t * curr_discrete[i];
		}

		float policy_grad[2 * SEGMENTS * 3];
		calc_policy_gradient(action, curr_discrete, &thread_theta[threadIdx.x * 2 * 3 * SEGMENTS], policy_grad);
		for (int i = 0; i < 2 * SEGMENTS * 3; i++) {
			thread_theta[threadIdx.x * 2 * SEGMENTS * 3 + i] += 0.01f * delta_t * policy_grad[i];
		}

		ep_reward += reward;
	}

	thread_rewards[threadIdx.x] = ep_reward;

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = 1; i < PARALLEL; i++) {
			for (int j = 0; j < 2 * SEGMENTS; j++) {
				thread_w[j] += thread_w[i * 2 * SEGMENTS + j];
			}
		}

		for (int i = 1; i < PARALLEL; i++) {
			for (int j = 0; j < 2 * SEGMENTS * 3; j++) {
				thread_theta[j] += thread_theta[i * 2 * SEGMENTS * 3 + j];
			}
		}

		for (int i = 0; i < 2 * SEGMENTS; i++) {
			w[i] = thread_w[i] / PARALLEL;
		}

		for (int j = 0; j < 2 * SEGMENTS * 3; j++) {
			theta[j] = thread_theta[j] / PARALLEL;
		}

		int total_reward = 0;
		for (int i = 0; i < PARALLEL; i++) {
			total_reward += thread_rewards[i];
		}
		*average_reward = total_reward / (int)PARALLEL;
		printf("average ep reward: %d\n", *average_reward);
	}
}

__global__ void cuda_act(float* curr_state, int* out, curandState* rand_state, float* theta) // good
{
	float discrete_state[200];
	segment_state(curr_state, discrete_state);

	float policy[3];
	calc_policy(discrete_state, theta, policy);

	// No built in way to sample from a discrete distribution in CUDA so need to do this manually
	float value = curand_uniform(rand_state);
	float action_0_low = 0;
	float action_0_high = policy[0];
	float action_1_low = action_0_high;
	float action_1_high = policy[1] + action_0_high;
	float action_2_low = action_1_high;
	float action_2_high = 1;

	if (value >= action_0_low && value <= action_0_high) {
		*out = 0;
	}
	else if (value >= action_1_low && value <= action_1_high) {
		*out = 1;
	}
	else if (value >= action_2_low && value <= action_2_high) {
		*out = 2;
	}
}

void agent_init() {
	cudaMalloc((void**)&w, sizeof(float) * 2 * SEGMENTS);
	cudaMemset(w, 0, sizeof(float) * 2 * SEGMENTS);

	cudaMalloc((void**)&theta, sizeof(float) * 2 * SEGMENTS * 3);
	cudaMemset(theta, 0, sizeof(float) * 2 * SEGMENTS * 3);

	cudaMalloc((void**)&d_action, sizeof(int));
	cudaMalloc((void**)&curr_state_act, sizeof(float) * 2);
	cudaMalloc((void**)&rand_state, sizeof(curandState));

	cudaMalloc((void**)&states_vector, sizeof(float) * 2 * 200 * PARALLEL);
	cudaMalloc((void**)&actions_vector, sizeof(int) * 200 * PARALLEL);
	cudaMalloc((void**)&rewards_vector, sizeof(int) * 200 * PARALLEL);
	cudaMalloc((void**)&next_states_vector, sizeof(float) * 2 * 200 * PARALLEL);
	cudaMalloc((void**)&average_reward, sizeof(int));

	cuda_agent_init<<<1, 1>>>(rand_state);
}

int agent_update(const float* host_states_vector, const int* host_actions_vector, const int* host_rewards_vector, const float* host_next_states_vector) {
	cudaMemcpy(states_vector, host_states_vector, sizeof(float) * 2 * 200 * PARALLEL, cudaMemcpyHostToDevice);
	cudaMemcpy(actions_vector, host_actions_vector, sizeof(int) * 200 * PARALLEL, cudaMemcpyHostToDevice);
	cudaMemcpy(rewards_vector, host_rewards_vector, sizeof(int) * 200 * PARALLEL, cudaMemcpyHostToDevice);
	cudaMemcpy(next_states_vector, host_next_states_vector, sizeof(float) * 2 * 200 * PARALLEL, cudaMemcpyHostToDevice);

	cuda_update <<<1, PARALLEL >>> (states_vector, actions_vector, rewards_vector, next_states_vector, w, theta, average_reward);

	return 0;

	int ret;
	cudaMemcpy(&ret, average_reward, sizeof(int), cudaMemcpyDeviceToHost);
	return ret;
}

int agent_act(const float* curr_state) {
	int h_action = -1;

	cudaMemcpy((void*)curr_state_act, (const void*)curr_state, sizeof(float)*2, cudaMemcpyHostToDevice);
	cuda_act<<<1, 1 >>> (curr_state_act, d_action, rand_state, theta);
	cudaMemcpy(&h_action, d_action, sizeof(int), cudaMemcpyDeviceToHost);

	return h_action;
}

void agent_free() {
	cudaFree(w);
	cudaFree(theta);
	cudaFree(d_action);
	cudaFree(curr_state_act);
	cudaFree(rand_state);
	cudaFree(states_vector);
	cudaFree(actions_vector);
	cudaFree(rewards_vector);
	cudaFree(next_states_vector);
}

int main() {
	agent_init();

	float* state = (float*)calloc(2 * 200 * PARALLEL, sizeof(float));
	int* actions = (int*)calloc(200 * PARALLEL, sizeof(int));
	int* rewards = (int*)calloc(200 * PARALLEL, sizeof(int));
	float* next = (float*)calloc(2 * 200 * PARALLEL, sizeof(float));

	agent_update(state, actions, rewards, next);

	agent_free();
}