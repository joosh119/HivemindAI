#include "CircularBuffer.h"
#include "NeuralNetwork.h"
#include <list>
#include <vector>
#include <algorithm>

#if __has_include(<emscripten/emscripten.h>)
    #include <emscripten/emscripten.h>
    #define PRINT(str) EM_ASM( console.log(str) );
    #define PRINT_VAR(var) EM_ASM( console.log(#var, $0), var);
#else
    #include <iostream>
    #define PRINT(str) std::cout << str;
    #define PRINT_VAR(var) std::cout << #var << ' ' << var;
#endif



// Network sizes (probably shouldn't be here)
const int AI_MAP_VISIBILITY = 9; // This is how many downward raycasts to perform around the enemy
const int INPUT_LAYER_SIZE = 7 + AI_MAP_VISIBILITY;
const int POLICY_OUTPUT_LAYER_SIZE = (2) * 2;

const int POLICY_DEFAULT_HIDDEN_LAYER_COUNT = 2;
const int VALUE_DEFAULT_HIDDEN_LAYER_COUNT = 2;
const int POLICY_DEFAULT_HIDDEN_LAYER_SIZE = 30;
const int VALUE_DEFAULT_HIDDEN_LAYER_SIZE = 30;

// Misc learning constants
const float REWARD_REDUCTION = 0.99f;
const float SMOOTHING_FACTOR = 0.95f;
const int MAX_VALUE_ITR = 3; // Number of epochs
const int MAX_POLICY_ITR = 3; // Number of epochs
const int BATCH_SIZE = 400; //// This large value means there will essentially be no minibatching
const float CLIP = 0.2f;
const float MIN_CLIP = 1.0f - CLIP;
const float MAX_CLIP = 1.0f + CLIP;
const float ENTROPY_BONUS = -10e-3f;
const float LEARNING_RATE = 3e-3f;
const float POLICY_REGULARIZATION_STRENGTH = 0e-5;
const float VALUE_REGULARIZATION_STRENGTH = 0e-5;
const float NOISE_SMOOTHING = 0.0f;
const float THRESHOLD_NOISE_CUTOFF = 0.5f;

// Max timestates saved in an episode
const int MAX_SAVED_TIME_STATES = 60*30; // 40 sec of saved timesteps with 30 fps



// Structs representing an episode for one enemy
class TimeState{
    public:
    TimeState() : state_v(std::vector<float>(INPUT_LAYER_SIZE)), actions_v(std::vector<float>(POLICY_OUTPUT_LAYER_SIZE)), choices_v(std::vector<float>(POLICY_OUTPUT_LAYER_SIZE/2)){
        time = 0;
        reward = 0;
    }
    
    public:
    float time;
    std::vector<float> state_v;
    std::vector<float> actions_v;
    float reward;
    std::vector<float> choices_v;
    public:
    float* getStateData() {
        return state_v.data();
    }
    float* getActionData() {
        return actions_v.data();
    }
    float& getRewardData() {
        return reward;
    }
    float* getChoicesData() {
        return choices_v.data();
    }
    const float* getStateData() const{
        return state_v.data();
    }
    const float* getActionData() const{
        return actions_v.data();
    }
    const float& getRewardData() const{
        return reward;
    }
    const float* getChoicesData() const{
        return choices_v.data();
    }
};
class Episode{
    private:
    CircularBuffer<TimeState> timestates;

    public:
    Episode() : timestates( MAX_SAVED_TIME_STATES ) {}
    Episode(const std::vector<Episode>& episodes) : timestates( sumEpisodeSizes(episodes) ){
        // Collect all episodes into a larger episode
        for(int i = 0; i < episodes.size(); ++i){
            for(int j = 0; j < episodes[i].size(); ++j){
                timestates.push_back(episodes[i].timestates[j]);
            }
        }
    }
    
    TimeState& addModifableTimeState(const float time){
        timestates.push_back(TimeState());
        timestates.back().time = time;
        return timestates.back();
    }
    void addRewardToBack(const float reward){
        timestates.back().getRewardData() += reward;
    }
    const CircularBuffer<TimeState>& getTimeStates() const{
        return timestates;
    }
    const TimeState& back() const{
        return timestates.back();
    }
    int size() const{
        return timestates.size();
    }

    private:
    static int sumEpisodeSizes(const std::vector<Episode>& episodes){
        int sum = 0;
        for(int i = 0; i < episodes.size(); ++i){
            sum += episodes[i].size();
        }
        return sum;
    }
};


// PPO
class PPO{
    public:
    // Network that estimates the reward
    NeuralNetwork value_network; 
    // Network that estimates the optimal movement
    NeuralNetwork policy_network;
    
    public:
    PPO(){ }
    // Initialize the networks with the default weights
    PPO(    const int input_size, const int policy_output_size,
            const int value_hidden_layer_size, const int value_hidden_layer_count,
            const int policy_hidden_layer_size, const int policy_hidden_layer_count)

        :   value_network( NeuralNetwork( input_size, 1, value_hidden_layer_size, value_hidden_layer_count, LEARNING_RATE, NeuralNetwork::RELU, NeuralNetwork::LINEAR)),
            policy_network(NeuralNetwork( input_size, policy_output_size, policy_hidden_layer_size, policy_hidden_layer_count, LEARNING_RATE)){
        
        // Set LINEAR for mean outputs and EXP for log stddev outputs (to get normal stddev)
        std::vector<NeuralNetwork::ActivationFunction> output_fns(policy_output_size);
        for(int i = 0; i < policy_output_size/2; ++i)
            output_fns[i] = NeuralNetwork::LINEAR;
        for(int i = policy_output_size/2; i < policy_output_size; ++i)
            output_fns[i] = NeuralNetwork::POW_1P1;
        policy_network.setActivationFunctions(NeuralNetwork::RELU, output_fns);
    }

    // Randomizes the value and policy network using Kaiming initialization
    void initialize(){
        value_network.kaimingInit();
        policy_network.kaimingInit();
    }


    // Computes the advantages for each timestep
    void computeAdvantages(const Episode& episode, float* advantages) const{
        const CircularBuffer<TimeState>& timestates = episode.getTimeStates();
        const int timestate_count = timestates.size();

        // Store output value for the one output node
        float out[1];

        // Calculate augmented rewards for each timestate, and then find the squared difference in values to the total error
        float timestep_reward = 0;

        // Find value for last timetstep
        const int T = timestate_count-1;
        value_network.propagate(timestates[T].state_v.data(), out);
        const float value = (*out);
        advantages[T] = timestates[T].reward - value;
        timestep_reward += value;

        for(int t = T-1; t >=0; --t){
            // Find reward guess by value network
            value_network.propagate(timestates[t].state_v.data(), out);
            const float value = (*out);
            
            // Find augmented reward for this timestep, and save it for previous step
            timestep_reward = timestates[t].reward + timestep_reward*REWARD_REDUCTION;
            advantages[t] = timestep_reward - value;
        }
    }
    // Computes the advantages for each timestep using the smoothing factor, and the targets for the value function
    void computeAdvantagesWithSmoothing(const Episode& episode, float* advantages, float* value_targets) const{
        const CircularBuffer<TimeState>& timestates = episode.getTimeStates();
        const int timestate_count = timestates.size();

        // Store values from one timestep ahead
        float future_value = 0;
        float future_advantage = 0;

        for(int t = timestate_count-1; t >=0; --t){
            // Find reward guess by value network
            float value;
            value_network.propagate(timestates[t].state_v.data(), &value);
            
            // Find augmented reward for this timestep, and save it for previous step
            const float delta = timestates[t].reward  +  (REWARD_REDUCTION*future_value)  -  value;
            advantages[t] = delta  +  (REWARD_REDUCTION*SMOOTHING_FACTOR)*future_advantage;
            
            // Add the value and advantages for the value target
            value_targets[t] = advantages[t] + value;

            // Set future values
            future_value = value;
            future_advantage = advantages[t];
        }
    }
    // Compute the discounted reward for each timestep
    void computeDiscountedRewards(const Episode& episode, float* discounted_rewards){
        const CircularBuffer<TimeState>& timestates = episode.getTimeStates();
        const int timestate_count = timestates.size();

        float prev_reward = 0;
        for(int t = timestate_count-1; t >=0; --t){
            prev_reward = timestates[t].reward + REWARD_REDUCTION*prev_reward;
            discounted_rewards[t] = prev_reward;
        }
    }


    // Trains the value network and returns the loss
    void trainValueNetwork(const Episode& episode, const float* value_target){
        const CircularBuffer<TimeState>& timestates = episode.getTimeStates();
        const int timestate_count = timestates.size();

        // Store values for each node
        const int node_count = value_network.getPropagateNodeCount();
        float prev_node_vals[node_count];
        float post_node_vals[node_count];
        // Create gradients to increment for each timestep
        const int weight_count = value_network.getWeightCount();
        const int bias_count = value_network.getBiasCount();
        float total_weight_grad[weight_count];
        float total_bias_grad[bias_count];
        
        // Create vector with timestep indexes
        std::vector<int> timestep_indexes = std::vector<int>(timestate_count);
        for(int l = 0; l < timestate_count; ++l)
            timestep_indexes[l] = l;
        
        // Do gradient ascent for a few steps
        for(int k = 0; k < MAX_VALUE_ITR; ++k){
            
            // Randomize timestate indexes
            std::default_random_engine gen(time(0));
            std::shuffle(timestep_indexes.begin(), timestep_indexes.end(), gen);

            int l = 0;
            while(l < timestate_count){
            
                // Initialize total weights at 0
                for(int i = 0; i < weight_count; i++)
                    total_weight_grad[i] = 0;
                for(int i = 0; i < bias_count; i++)
                    total_bias_grad[i] = 0;

                int c = 0;
                for(; c < BATCH_SIZE  &&  l < timestate_count; ++l, ++c){
                    const int t = timestep_indexes[l];
                    
                    // Find reward guess by value network
                    value_network.propagateStore(timestates[t].state_v.data(), prev_node_vals, post_node_vals);
                    const float value = post_node_vals[node_count-1];
                    // Calculate loss derivative
                    const float loss_D = -2 * (value_target[t] - value);
                    // Find weights for current values
                    float weight_grad[weight_count];
                    float bias_grad[bias_count];
                    value_network.backpropagateStore(timestates[t].state_v.data(), prev_node_vals, post_node_vals, &loss_D, weight_grad, bias_grad);

                    // Add weights to totals
                    for(int i = 0; i < weight_count; ++i)
                        total_weight_grad[i] += weight_grad[i];
                    for(int i = 0; i < bias_count; ++i)
                        total_bias_grad[i] += bias_grad[i];
                }

                // Average total weights
                for(int i = 0; i < weight_count; ++i)
                    total_weight_grad[i] /= c;
                for(int i = 0; i < bias_count; ++i)
                    total_bias_grad[i] /= c;
                // Account for regularization in gradient
                value_network.addRegularization(total_weight_grad, total_bias_grad, VALUE_REGULARIZATION_STRENGTH); 
                // Apply gradients
                value_network.applyGradients(total_weight_grad, total_bias_grad, -1);
            }
        }
    }
    
    // Trains the policy network
    void trainPolicyNetwork(const Episode& episode, const float* advantages){
        const CircularBuffer<TimeState>& timestates = episode.getTimeStates();
        const int timestate_count = timestates.size();

        // Store starting index for outputs
        const int output_index = policy_network.getBiasLayerIndex(policy_network.getHiddenLayerCount());

        // Create gradients to increment for each timestep
        const int weight_count = policy_network.getWeightCount();
        const int bias_count = policy_network.getBiasCount();
        float total_weight_grad[weight_count];
        float total_bias_grad[bias_count];
        // Store weights
        const int node_count = policy_network.getPropagateNodeCount();
        float prev_node_vals[node_count];
        float post_node_vals[node_count];

        // Store probabilites for old policy
        const int output_count = policy_network.getOutputCount();   
        const int half_output_count = output_count/2;   
        float prob_old[timestate_count][half_output_count];
        for(int t = 0; t < timestate_count; ++t){
            for(int i = 0; i < half_output_count; ++i){
                prob_old[t][i] = probabilityDensity(timestates[t].actions_v[i], timestates[t].actions_v[half_output_count + i], timestates[t].choices_v[i]);
            }
        }

        // Create vector with timestep indexes
        std::vector<int> timestep_indexes = std::vector<int>(timestate_count);
        for(int l = 0; l < timestate_count; ++l)
            timestep_indexes[l] = l;


        // Do gradient ascent for a few steps
        for(int k = 0; k < MAX_POLICY_ITR; ++k){

            // Randomize timestate indexes
            std::default_random_engine gen(time(0));
            std::shuffle(timestep_indexes.begin(), timestep_indexes.end(), gen);

            // For each time we iterate on the episode, update the gradient a few times
            int l = 0;
            while(l < timestate_count){
                
                // Initialize total weights at 0
                for(int i = 0; i < weight_count; i++)
                    total_weight_grad[i] = 0;
                for(int i = 0; i < bias_count; i++)
                    total_bias_grad[i] = 0;

                // Iterate through each timestep
                int c = 0;
                for(; c < BATCH_SIZE  &&  l < timestate_count; ++l, ++c){
                    const int t = timestep_indexes[l];

                    // Find weights for current values
                    float weight_grad[weight_count];
                    float bias_grad[bias_count];

                    // Get outputs for the new policy
                    policy_network.propagateStore(timestates[t].state_v.data(), prev_node_vals, post_node_vals);
                    float *out = post_node_vals + output_index;

                    // Find probability of this action with the updated policy
                    float prob_new[half_output_count];
                    float r[half_output_count];
                    for(int i = 0; i < half_output_count; ++i){
                        prob_new[i] = probabilityDensity(out[i], out[half_output_count + i], timestates[t].choices_v[i]);
                        r[i] = prob_new[i] / prob_old[t][i]; 
                    }

                    // Find the gradient of the clipped loss function
                    float loss_D[output_count];
                    for(int i = 0; i < half_output_count; ++i){ 
                        // Clip probability ratio
                        if(advantages[t] >= 0  &&  prob_new[i] >= MAX_CLIP*prob_old[t][i]){
                            loss_D[i] =                     0;// Mean
                            loss_D[half_output_count + i] =    0;// Standard deviation
                        }
                        else if(advantages[t] < 0  &&  prob_new[i] <= MIN_CLIP*prob_old[t][i]){
                            loss_D[i] =                     0; // Mean
                            loss_D[half_output_count + i] =    0; // Standard deviation
                        }

                        // If no clip, do normal derivative
                        else{
                            const float mean =  out[i];
                            const float stddev = out[half_output_count + i];
                            const float stddev_2 = stddev*stddev;
                            const float mean_diff = timestates[t].choices_v[i] - mean;

                            // Mean
                            loss_D[i] =                     (advantages[t]) * (r[i]) * (mean_diff / stddev_2);
                            // Standard deviation: Loss + entropy
                            loss_D[half_output_count + i] =    (advantages[t]) * (r[i]) * (((mean_diff*mean_diff) / (stddev_2*stddev)) - (1/stddev))
                                                            + ENTROPY_BONUS * (1/stddev);
                        }
                    }

                    // Backpropagate with the given loss derivative
                    policy_network.backpropagateStore(timestates[t].state_v.data(), prev_node_vals, post_node_vals, loss_D, weight_grad, bias_grad);

                    // Add weights to totals
                    for(int i = 0; i < weight_count; ++i)
                        total_weight_grad[i] += weight_grad[i];
                    for(int i = 0; i < bias_count; ++i)
                        total_bias_grad[i] += bias_grad[i];
                
                }   // End minibatch

                // Average total weights
                for(int i = 0; i < weight_count; ++i)
                    total_weight_grad[i] /= c;
                for(int i = 0; i < bias_count; ++i)
                    total_bias_grad[i] /= c;

                // Account for regularization in gradient
                policy_network.addRegularization(total_weight_grad, total_bias_grad, -POLICY_REGULARIZATION_STRENGTH); 
                // Apply gradients
                policy_network.applyGradients(total_weight_grad, total_bias_grad, 1);
            }
        } // End episode
    }
    
    // Trains the AI with the given episode
    void train(const Episode& episode){
        const int episode_size = episode.getTimeStates().size();
        // Find advantages
        float advantages[episode_size];
        float _value_targets[episode_size];
        computeAdvantagesWithSmoothing(episode, advantages, _value_targets);

        // Find discounted rewards
        float discounted_rewards[episode_size];
        computeDiscountedRewards(episode, discounted_rewards);

        // Train both networks
        trainValueNetwork(episode, discounted_rewards);
        trainPolicyNetwork(episode, advantages);
    }

    // Trains the AI with all of the given episodes
    void train(const std::vector<Episode>& episodes){
        if(episodes.size() == 0) 
            return;
        // Just train on the 1 episode if only one exists
        if(episodes.size() == 1){
            this->train(episodes[0]);
            return;
        }
        // Else train on every episode at once, to avoid overprioritizing the first trajectory

        // Push all episodes into one longer episode
        const Episode all_episodes(episodes);

        // Calculate all advantages and discounted rewards for each episode, and place them into one buffer
        float all_advantages[all_episodes.size()];
        float all_discounted_rewards[all_episodes.size()];
        float _value_targets[all_episodes.size()];
        int s_index = 0;
        for(int i = 0; i < episodes.size(); ++i){
            computeAdvantagesWithSmoothing(episodes[i], all_advantages + s_index, _value_targets + s_index);
            computeDiscountedRewards(episodes[i], all_discounted_rewards + s_index);

            s_index += episodes[i].size();
        }

        // The indexes for the advantages and timesteps in all_episodes should correspond with each other at this point
        // Now train on the one big episode
        trainValueNetwork(all_episodes, all_discounted_rewards);
        trainPolicyNetwork(all_episodes, all_advantages);
    }


    // Choose outputs given the mean and stddev from the ouputs
    // The choice is just the value chosen directly from the mean and stddev, and the aug_choice
    #define PURE_NOISE_SMOOTHING
    void sampleChoice(const float* outputs, float* prev_smoothed_outputs, float* prev_noise, float* choices, float* aug_choices, std::mt19937& gen, const float min = -1, const float max = 1){
        const float range = max - min;
        
        const int half_output_count = policy_network.getOutputCount()/2;
        for(int i = 0; i < half_output_count; ++i){
            // Get raw noise from distribution
            const float mean = outputs[i];
            const float stddev = outputs[half_output_count + i];

            #if defined(PURE_NOISE_SMOOTHING)
                // Cutoff noise if mean or stddev is too different than previous
                if( std::abs(mean - prev_smoothed_outputs[i]) > THRESHOLD_NOISE_CUTOFF ||  std::abs(stddev - prev_smoothed_outputs[half_output_count + i]) > THRESHOLD_NOISE_CUTOFF ){
                    prev_noise[i] = 0;
                }
                prev_smoothed_outputs[i] = mean;
                prev_smoothed_outputs[half_output_count + i] = stddev;
                // Get raw noise from deviation
                std::normal_distribution<float> dst(0.0f, stddev);
                const float raw_noise = dst(gen);
                // Smooth noise
                const float smooth_noise = NOISE_SMOOTHING*prev_noise[i]  +  (1-NOISE_SMOOTHING)*raw_noise;
                prev_noise[i] = smooth_noise;
                // Add sampled noise to mean
                choices[i] = mean + smooth_noise;
            #else
                // Smooth mean and stddev
                float smoothed_mean = NOISE_SMOOTHING*prev_smoothed_outputs[i]  +  (1-NOISE_SMOOTHING)*mean;
                float smoothed_stddev = NOISE_SMOOTHING*prev_smoothed_outputs[half_output_count + i]  +  (1-NOISE_SMOOTHING)*stddev;
                prev_smoothed_outputs[i] = smoothed_mean;
                prev_smoothed_outputs[half_output_count + i] = smoothed_stddev;
                // Get raw noise from smoothed deviation
                std::normal_distribution<float> dst(0.0f, smoothed_stddev);
                const float raw_noise = dst(gen);
                // Smooth noise
                const float smoothed_noise = NOISE_SMOOTHING*prev_noise[i]  +  (1-NOISE_SMOOTHING)*raw_noise;
                prev_noise[i] = smoothed_noise;
                // Add sampled noise to mean
                choices[i] = smoothed_mean + smoothed_noise;
            #endif

            // Transformation sample to fit within range
            aug_choices[i] = 0.5f*(min + max + std::tanh(choices[i])*range);
        }
    }

    // Returns the probability density at the mean of a normal distribution
    static float probabilityDensity(const float mean, const float stdev, const float choice){
        // If the stdev is 0, then the probability function approaches 0
        // This avoids Nan results, since inf*0 = nan
        if(stdev <= 1e-5)
            return 0;
        
        // 1/(stdev*sqrt(2*PI))
        const float i_std = (1.0f/2.5066283f) * (1.0f/stdev);
        // e^(-(choice - mean)^2/(2stdev^2))
        const float cmm = choice - mean;
        const float e_p = std::exp(-(cmm*cmm) / (2*stdev*stdev));
        
        return i_std * e_p;
    }
};