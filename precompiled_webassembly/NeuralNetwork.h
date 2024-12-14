#include <cmath>
#include <vector>
#include <random>
#include <ctime>   // for time()
#include <iostream>


const float N_EPSILON = 1e-7;

inline float randomRange(const float min = 0, const float max = 1){
    return ((double) std::rand() / RAND_MAX)*(max-min) + min;
}


class NeuralNetwork{
    public:
    // Struct that stores an activation function and it's derivative
    struct ActivationFunction{
        float (*fn) (const float);
        float (*fn_d) (const float);
    };
    struct LossFunction{
        float (*fn) (const float*, const float*, const int);
        float (*fn_d) (const float);
    };


    private:
    // Variables to set directly from JS
    int input_layer_size;
    int output_layer_size;
    int hidden_layer_size;
    int hidden_layer_count;
    std::vector<float> weights;
    std::vector<float> bias;

    ActivationFunction hidden_fn;
    std::vector<ActivationFunction> output_fns;


    public:
    // Activation functions
    constexpr static ActivationFunction LINEAR = {
        [](const float v){
            return v;
        },
        [](const float v){
            return 1.0f;
        }
    };
    constexpr static ActivationFunction RELU = {
        [](const float v){
            return std::max(0.0f, v);
        },
        [](const float v){
            return (float) !std::signbit(v);
        }
    };
    constexpr static ActivationFunction SIGMOID = { // Sigmoid from 0 to 1
        [](const float v){
            return 1.0f/(1+std::exp(-v));
        },
        [](const float v){
            const float sigmoid = 1.0f/(1+std::exp(-v));
            return sigmoid*(1-sigmoid);
        }
    };
    constexpr static ActivationFunction AUG_SIGMOID = { // Sigmoid from -1 to 1
        [](const float v){
            return 2.0f/(1+std::exp(-v)) - 1.0f;
        },
        [](const float v){
            const float aug_sigmoid = 2.0f/(1+std::exp(-v)) - 1.0f;
            return (aug_sigmoid+1)*(1-aug_sigmoid)/2;
        }
    };
    constexpr static ActivationFunction TANH = {
        [](const float v){
            return std::tanh(v);
        },
        [](const float v){
            const float tanh = std::tanh(v);
            return 1 - tanh*tanh;
        }
    };
    constexpr static ActivationFunction EXP = {
        [](const float v){
            return std::exp(v);
        },
        [](const float v){
            return std::exp(v);
        }
    };
    constexpr static ActivationFunction POW_1P1 = {
        [](const float v){
            return std::pow(1.1f, v);
        },
        [](const float v){
            return 0.09531018f * std::pow(1.1f, v);
        }
    };

    // Loss functions
    constexpr static LossFunction MEAN_SQUARED = {
        [](const float* v, const float* actual, const int c){
            float sqr_sum = 0;
            for(int i = 0; i < c; ++i){
                const float diff = actual[i] - v[i];
                sqr_sum += diff*diff;
            }
            return sqr_sum/c;
        },
        [](const float v){
            return -2*v;
        }
    };


    public:
    NeuralNetwork() {}
    NeuralNetwork(  const int input_layer_size, const int output_layer_size, const int hidden_layer_size, const int hidden_layer_count,
                    const ActivationFunction& hidden_fn = RELU, const ActivationFunction& output_fn = SIGMOID){
        // Set sizes
        this->input_layer_size = input_layer_size;
        this->output_layer_size = output_layer_size;
        this->hidden_layer_size = hidden_layer_size;
        this->hidden_layer_count = hidden_layer_count;
        // Set activation functions
        this->hidden_fn = hidden_fn;
        this->output_fns = std::vector<ActivationFunction>(output_layer_size, output_fn);
        // Calculate total nodes and edges
        recalculate();
    }
    
    // Recalculates needed values upon direct value changes
    // Sets all weights and bias' to 0
    void recalculate(){
        const int weight_count = input_layer_size*hidden_layer_size  +  hidden_layer_size*hidden_layer_size*(hidden_layer_count-1)  +  output_layer_size*hidden_layer_size;
        const int bias_count = hidden_layer_size*hidden_layer_count + output_layer_size;

        // Initialize all weights at zero
        weights = std::vector<float>(weight_count, 0);
        bias = std::vector<float>(bias_count, 0);
    }
    // Sets random weights between min and max
    void randomInit(const float min = -1, const float max = 1){
        for(int i = 0; i < weights.size(); i++){
            weights[i] = randomRange(min, max);
        }
        for(int i = 0; i < bias.size(); i++){
            bias[i] = 0.0f;
        }
    }
    // Sets random weights based on Kaiming initialization
    void kaimingInit(){
        std::random_device rd;
        std::mt19937 rd_gen(rd());
        std::normal_distribution dst(0.0f, 1.0f);

        const int input_weight_end_index = input_layer_size*hidden_layer_size;
        for(int i = 0; i < weights.size(); i++){
            int n_in = hidden_layer_size;
            if(i < input_weight_end_index)
                n_in = input_layer_size;

            weights[i] = dst(rd_gen) * std::sqrt(2.0f/n_in);
        }
        for(int i = 0; i < bias.size(); i++){
            bias[i] = 0;
        }
    }
    // Sets the activation functions of the network
    void setActivationFunctions(const ActivationFunction& hidden_fn, const std::vector<ActivationFunction>& output_fns){
        this->hidden_fn = hidden_fn;
        this->output_fns = output_fns;
    }

    // Propagate the signals from the inputs into the outputs, and store each value at each node, before and after the activation function
    void propagateStore(const float* inputs, float* prev_node_vals, float* post_node_vals) const{          
        // The starting index of the weights pointing into a given node
        int weight_start_index = 0;

        // Propagate from input layer
        {const int this_layer_start_index = this->getBiasLayerIndex(0);
        for(int i = 0; i < hidden_layer_size; i++){
            const int node_index = this_layer_start_index + i;

            prev_node_vals[node_index] = bias[node_index];
            for(int j = 0; j < input_layer_size; j++){
                prev_node_vals[node_index] += inputs[j] * weights[weight_start_index + j];
            }

            post_node_vals[node_index] = hidden_fn.fn(prev_node_vals[node_index]);

            weight_start_index += input_layer_size;
        }}

        // Propagate between hidden layers
        for(int k = 1; k < hidden_layer_count; k++){
            
            const int this_layer_start_index = this->getBiasLayerIndex(k);
            const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
            for(int i = 0; i < hidden_layer_size; i++){
                const int node_index = this_layer_start_index + i;

                prev_node_vals[node_index] = bias[node_index];
                for(int j = 0; j < hidden_layer_size; j++){
                    prev_node_vals[node_index] += post_node_vals[prev_layer_start_index + j] * weights[weight_start_index + j];
                }

                post_node_vals[node_index] = hidden_fn.fn(prev_node_vals[node_index]);

                weight_start_index += hidden_layer_size;
            }
        }

        // Propagate to output layer
        {const int this_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
        const int prev_layer_start_index = this->getBiasLayerIndex(hidden_layer_count-1);
        for(int i = 0; i < output_layer_size; i++){
            const int node_index = this_layer_start_index + i;

            prev_node_vals[node_index] = bias[node_index];
            for(int j = 0; j < hidden_layer_size; j++){
                prev_node_vals[node_index] += post_node_vals[prev_layer_start_index + j] * weights[weight_start_index + j];
            }

            post_node_vals[node_index] = output_fns[i].fn( prev_node_vals[node_index] );

            weight_start_index += hidden_layer_size;
        }}
    }
    // Propagate the signals from the inputs into the outputs
    void propagate(const float* inputs, float* outputs) const{
        // Arrays to hold the current layers computed values
        float hidden_node_val[hidden_layer_size];
        float prev_hidden_node_val[hidden_layer_size];

        // The starting index of the weights pointing into a given node
        int weight_start_index = 0;

        // Propagate from input layer
        {const int this_layer_start_index = this->getBiasLayerIndex(0);
        for(int i = 0; i < hidden_layer_size; ++i){
            const int node_index = this_layer_start_index + i;

            hidden_node_val[i] = bias[node_index];
            for(int j = 0; j < input_layer_size; j++){
                hidden_node_val[i] += inputs[j] * weights[weight_start_index + j];
            }

            hidden_node_val[i] = hidden_fn.fn(hidden_node_val[i]);

            weight_start_index += input_layer_size;
        }}

        // Propagate between hidden layers
        for(int k = 1; k < hidden_layer_count; ++k){
            // Save previous hidden node values before setting new ones
            for(int i = 0; i < hidden_layer_size; ++i){
                prev_hidden_node_val[i] = hidden_node_val[i];
            }

            const int this_layer_start_index = this->getBiasLayerIndex(k);
            const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
            for(int i = 0; i < hidden_layer_size; ++i){
                const int node_index = this_layer_start_index + i;

                hidden_node_val[i] = bias[node_index];
                for(int j = 0; j < hidden_layer_size; ++j){
                    hidden_node_val[i] += prev_hidden_node_val[j] * weights[weight_start_index + j];
                }

                hidden_node_val[i] = hidden_fn.fn(hidden_node_val[i]);

                weight_start_index += hidden_layer_size;
            }
        }

        // Propagate to output layer
        const int this_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
        const int prev_layer_start_index = this->getBiasLayerIndex(hidden_layer_count-1);
        for(int i = 0; i < output_layer_size; i++){
            const int node_index = this_layer_start_index + i;

            outputs[i] = bias[node_index];
            for(int j = 0; j < hidden_layer_size; j++){
                outputs[i] += hidden_node_val[j] * weights[weight_start_index + j];
            }

            outputs[i] = output_fns[i].fn( outputs[i] );

            weight_start_index += hidden_layer_size;
        }
    }
    
    // Find the gradients with the squared loss and node values obtained from propagateStore() and store the gradients
    // The output_D is essentially the derivative of the bias' for the output nodes
    void backpropagateStore(const float* inputs, const float* prev_node_vals, const float* post_node_vals, const float* output_D,
                            float* weight_grad, float* bias_grad) const{
        float weight_sums[hidden_layer_size];
        float prev_weight_sums[hidden_layer_size];
        for(int j = 0; j < hidden_layer_size; j++)
            weight_sums[j] = 0;
        
        // The starting index of the weights pointing into a given node
        int weight_start_index = this->getNodeWeightsIndex(bias.size()-1);

        // Find grads for output nodes
        {const int this_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
        const int prev_layer_start_index = this->getBiasLayerIndex(hidden_layer_count-1);
        for(int i = output_layer_size-1; i >= 0; --i){
            const int node_index = this_layer_start_index + i;
            // Bias
            bias_grad[node_index] = (1) * output_fns[i].fn_d(prev_node_vals[node_index]) * (output_D[i]) / output_layer_size;
            
            // Weights
            for(int j = hidden_layer_size-1; j >= 0; --j){
                weight_grad[weight_start_index + j] = post_node_vals[prev_layer_start_index + j] * bias_grad[node_index]; 
                weight_sums[j] += weights[weight_start_index + j] * bias_grad[node_index];
            }

            weight_start_index -= hidden_layer_size;
        }}

        // Find grads between hidden nodes
        for(int k = hidden_layer_count-1; k >= 1; --k){
            for(int j = 0; j < hidden_layer_size; j++){
                prev_weight_sums[j] = weight_sums[j];
                weight_sums[j] = 0;
            }

            const int this_layer_start_index = this->getBiasLayerIndex(k);
            const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
            for(int i = hidden_layer_size-1; i >= 0; --i){
                const int node_index = this_layer_start_index + i;
                // Bias
                bias_grad[node_index] = (1) * hidden_fn.fn_d(prev_node_vals[node_index]) * prev_weight_sums[i];

                // Weights
                for(int j = hidden_layer_size-1; j >= 0; --j){
                    weight_grad[weight_start_index + j] = post_node_vals[prev_layer_start_index+j] * bias_grad[node_index]; 
                    weight_sums[j] += weights[weight_start_index + j] * bias_grad[node_index];
                }
                
                weight_start_index -= hidden_layer_size;
            }
        }
        // Ensure start index accounts for difference in edges with the nodes in the first hidden layer
        weight_start_index += hidden_layer_size;
        weight_start_index -= input_layer_size;
        
        // Find grads for input nodes
        {const int this_layer_start_index = this->getBiasLayerIndex(0);
        for(int i = hidden_layer_size-1; i >= 0; --i){
            const int node_index = this_layer_start_index + i;
            // Bias
            bias_grad[node_index] = (1) * hidden_fn.fn_d(prev_node_vals[node_index]) * (weight_sums[i]);
            
            // Weights
            for(int j = input_layer_size-1; j >= 0; --j){
                weight_grad[weight_start_index + j] = inputs[j] * bias_grad[node_index]; 
            }
            
            weight_start_index -= input_layer_size;
        }}
    }
    // Find the gradients with the squared loss and node values obtained from propagateStore() and update the bias and weights
    void backpropagate(const float* inputs, const float* prev_node_vals, const float* post_node_vals, const float* output_D, const float learning_rate){
        float weight_grad[weights.size()];
        float bias_grad[bias.size()];
        
        backpropagateStore(inputs, prev_node_vals, post_node_vals, output_D, weight_grad, bias_grad);

        applyGradients(weight_grad, bias_grad, learning_rate);
    }

    // Update weights with gradients
    // Subtracts weights for gradient descent
    void applyGradients(const float* weight_grad, const float* bias_grad, const float learning_rate){
        for(int i = 0; i < weights.size(); i++){
            weights[i] -= weight_grad[i] * learning_rate;
        }
        for(int i = 0; i < bias.size(); i++){
            bias[i] -= bias_grad[i] * learning_rate;
        }
    }


    // Apply L2 regularization to gradients
    void addRegularization(float* weight_grad, float* bias_grad, const float regularization_strength) const{
        for(int i = 0; i < getWeightCount(); ++i){
            weight_grad[i] += 2 * weights[i] * regularization_strength;
        }
        for(int i = 0; i < getBiasCount(); ++i){
            bias_grad[i] += 2 * bias[i] * regularization_strength;
        }
    }

    // Get sizes for initializing external arrays
    inline int getOutputCount() const{
        return output_layer_size;
    }
    inline int getPropagateNodeCount() const{
        return bias.size();
    }
    inline int getWeightCount() const{
        return weights.size();
    }
    inline int getBiasCount() const{
        return bias.size();
    }
    inline int getHiddenLayerCount() const{
        return hidden_layer_count;
    }

    // Finds the start pointer for the weights in a given layer
    // Layer 0 contains edges between the input and first hidden layer
    inline int getWeightLayerIndex(const int layer) const{
        const int first_layer_edge_count = input_layer_size*hidden_layer_size;
        const int hidden_layer_edge_count = hidden_layer_size*hidden_layer_size;
        
        return (layer==0 ? 0 : first_layer_edge_count + hidden_layer_edge_count*(layer-1));
    }
    float* getWeightLayer(const int layer){
        const int index = getWeightLayerIndex(layer);
        return (&weights[0] + index);
    }
    // Finds the start pointer for the bias' in a given layer
    // Layer 0 is the first hidden layer and the layer (hidden_layer_count) is the output layer (because there is (hidden_layer_count+1) total layers)
    inline int getBiasLayerIndex(const int layer) const{
        return layer*hidden_layer_size;
    }
    float* getBiasLayer(const int layer){
        const int index = getBiasLayerIndex(layer);
        return (&bias[0] + index);
    }
    // Gets the start pointer for the weights pointing to the node at the given index
    inline int getNodeWeightsIndex(const int node_index) const{
        // Between input and first hidden layer
        if(node_index < hidden_layer_size)
            return node_index*input_layer_size;
        else
            return (hidden_layer_size*input_layer_size) + (node_index-hidden_layer_size)*hidden_layer_size;
    }
    float* getNodeWeights(const int node_index){
        const int index = getNodeWeightsIndex(node_index);
        return (&weights[0] + index);
    }
    // Gets the number of weight edges pointing into the given bias node
    inline int getNodeWeightCount(const int node_index){
        // Between input and first hidden layer
        if(node_index < hidden_layer_size)
            return input_layer_size;
        else
            return hidden_layer_size;
    }

    // Various debugging methods
    float getWeightAverage() const{
        float sum = 0;
        for(int i = 0; i < weights.size(); ++i){
            sum += weights[i];
        }
        return sum / weights.size();
    }
    float getBiasAverage() const{
        float sum = 0;
        for(int i = 0; i < bias.size(); ++i){
            sum += bias[i];
        }
        return sum / bias.size();
    }
    float getWeightDiff(const NeuralNetwork& other) const {
        if(weights.size() != other.weights.size()){
            throw ("Networks are not the same size!");
        }

        float sqrd_diff_sum;
        for(int i = 0; i < weights.size(); ++i){
            float diff = weights[i] - other.weights[i];
            sqrd_diff_sum += diff*diff;
        }
        return std::sqrt(sqrd_diff_sum);
    }
    float getBiasDiff(const NeuralNetwork& other) const {
        if(bias.size() != other.bias.size()){
            throw ("Networks are not the same size!");
        }

        float sqrd_diff_sum;
        for(int i = 0; i < bias.size(); ++i){
            float diff = bias[i] - other.bias[i];
            sqrd_diff_sum += diff*diff;
        }
        return std::sqrt(sqrd_diff_sum);
    }

    // For printing the network
    friend std::ostream& operator<< (std::ostream& fs, const NeuralNetwork& n){
        // Layer sizes
        fs << "Layers: " << n.input_layer_size << ' ';
        for(int i = 0; i < n.hidden_layer_count; i++)
            fs << n.hidden_layer_size << ' ';
        fs << n.output_layer_size;
        // Bias and weight values
        fs << "\nBias': ";
        for(int i = 0; i < n.bias.size(); i++){
            fs << n.bias[i] << ' ';
        }
        fs << "\nWeights: ";
        for(int i = 0; i < n.weights.size(); i++){
            fs << n.weights[i] << ' ';
        }

        return fs;
    }
};

