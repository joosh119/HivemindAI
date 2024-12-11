// compile.bat hivemind.cpp ..\Hivemind_Website\hivemind\wasm\wasm.js

#include <cassert>
#include "PPO.h"

#if __has_include(<emscripten/emscripten.h>)
    #include <emscripten/emscripten.h>
    #define PRINT(str) EM_ASM( console.log(str) );
    #define PRINT_VAR(var) EM_ASM( console.log(#var, $0), var);
#else
    #include <iostream>
    #define PRINT(str) std::cout << str;
    #define PRINT_VAR(var) std::cout << #var << ' ' << var;
#endif
//#define PRINT(str);


// Training mode
#define MANUAL_TRAIN      // Train the AI to kill the player, and not get killed
//#define TARGET_TRAIN      // Train the AI to target the stationary player
//#define MOVE_TRAIN          // Train the AI to move around as much as possible


// Constants
const float _EPSILON = 0.00001;
const float _INFINITY = 1.0/0.0;


// Environment constants
const int MAX_TERRAIN_COUNT = 100;
const int MAX_ENEMY_COUNT = 50;
const float BOUNDS_BUFFER_MULTIPLIER = 10;
const float VERTICAL_VISIBILITY_MULTIPLIER = 2;

#if defined( MANUAL_TRAIN )
const float MAX_SIMULATION_TIME = 90*1000; // Max time the simulation can run before resetting
const float MAX_TIME_IN_ENV_MS = 0*1000;
#elif defined( TARGET_TRAIN ) || defined(MOVE_TRAIN)
const float MAX_SIMULATION_TIME = 60*1000; // Max time the simulation can run before resetting
const float MAX_TIME_IN_ENV_MS = 0*1000;
#endif


// Neural Network constants
const int FPS = 30;
const float ACTION_REPEAT_TIME = 1000.0f / 7.5f; // Get policy every 133 ms (8 ticks at 30fps)
const int ACTION_REPEAT_TICKS = std::round(ACTION_REPEAT_TIME / FPS);

const float REWARD_MOD = 10;
// The maximum an accumulated reward can be is 1/(1-REWARD_REDUCTION) * reward.
// So, ensure that terminating state rewards are greater than that to ensure that reaching a positive terminal state is more rewarding than staying alive forever.
// The maximum an accumulated punishment can be is 1/(1-REWARD_REDUCTION) * punishment.
// So, ensure that terminating state punishment are greater than that to ensure that reaching a negative terminal state is more punishing than dying quickly.
#if defined( MANUAL_TRAIN )
const float ALIVE_REWARD =      ( 0.50  )   / REWARD_MOD; // Reward for staying alive 1 second is 1
const float MOVE_REWARD =       ( 0.00  )   / REWARD_MOD; // Reward for moving at max speed for 1 second is 0.5
const float DISTANCE_REWARD =   (-0.00  )   / REWARD_MOD; // Punishment for being at max distance for 1 second is -4
const float FALLING_REWARD =    (-0.00  )   / REWARD_MOD; // Punishment for falling for 1 second is -2
const float JUMPING_REWARD =    (-0.00  )   / REWARD_MOD; // Punishment for jumping is -0.5     // NOT IMPLEMENTED 
const float PLAYER_KILL_REWARD=   100.00    / REWARD_MOD;
const float FALL_DEATH_REWARD =  -50.00     / REWARD_MOD;
const float PLAYER_DEATH_REWARD= -10.00     / REWARD_MOD;

#elif defined(TARGET_TRAIN)
const float ALIVE_REWARD =      ( 0.00  )   / REWARD_MOD; // Reward for staying alive 1 second is 1
const float MOVE_REWARD =       ( 0.00  )   / REWARD_MOD; // Reward for moving at max speed for 1 second is 0.5
const float DISTANCE_REWARD =   (-1.00  )   / REWARD_MOD; // Punishment for being at max distance for 1 second is -4
const float FALLING_REWARD =    (-0.00  )   / REWARD_MOD; // Punishment for falling for 1 second is -2
const float JUMPING_REWARD =    (-0.00  )   / REWARD_MOD; // Punishment for jumping is -0.5     // NOT IMPLEMENTED 
const float PLAYER_KILL_REWARD=   0.00     / REWARD_MOD;
const float FALL_DEATH_REWARD =  -200.0     / REWARD_MOD;
const float PLAYER_DEATH_REWARD= -0.00      / REWARD_MOD;

#elif defined( MOVE_TRAIN )
const float ALIVE_REWARD =      ( 0.00  )   / REWARD_MOD; // Reward for staying alive 1 second is 1
const float MOVE_REWARD =       ( 1.00  )   / REWARD_MOD; // Reward for moving at max speed for 1 second is 0.5
const float DISTANCE_REWARD =   (-0.00  )   / REWARD_MOD; // Punishment for being at max distance for 1 second is -4
const float FALLING_REWARD =    (-0.00  )   / REWARD_MOD; // Punishment for falling for 1 second is -2
const float JUMPING_REWARD =    (-0.00  )   / REWARD_MOD; // Punishment for jumping is -0.5     // NOT IMPLEMENTED
const float PLAYER_KILL_REWARD=   0.00      / REWARD_MOD;
const float FALL_DEATH_REWARD =  -200.00    / REWARD_MOD;
const float PLAYER_DEATH_REWARD= -0.00      / REWARD_MOD;
#endif



// Structs
struct Vector2{
    float x,y;
};
struct Object{
    Vector2 pos;
    Vector2 size;
};

// Static Environment
struct Environment{
    int terrain_count = 0;
    int enemy_count = 0;
    Object player;
    Object terrain[MAX_TERRAIN_COUNT];
    Object enemies[MAX_ENEMY_COUNT]; 
};
struct NetworkPointers{
    const int input_size = INPUT_LAYER_SIZE;
    const int value_output_size = 1;
    const int policy_output_size = POLICY_OUTPUT_LAYER_SIZE;
    int value_hidden_layer_size;
    int value_hidden_layer_count;
    int policy_hidden_layer_size;
    int policy_hidden_layer_count;
    float* value_weights;
    float* value_bias;
    float* policy_weights;
    float* policy_bias;
};
struct GameMetadata{
    int player_wins = 0;
    int player_losses = 0;
    float env_time_ms = 0;
    float total_env_time_ms = 0;
    float total_time_ms = 0;
    float last_round_end_reward = 0;
    float enemy_rewards[MAX_ENEMY_COUNT];
};
struct EnvironmentConstants{
    float gravity;
    float drag;
    float move_acc;
    float jump_speed;
    float enemy_visibility;
    // Other calculated constants
    Vector2 min_bound;
    Vector2 max_bound;
    float max_move_speed;
    float max_jump_height;
    float max_fall_speed; // terminal velocity
};

// Static Environment Variables
Environment initial_environment;
Environment active_environment;
NetworkPointers network_ptrs;
GameMetadata game_metadata;

EnvironmentConstants env_constants;

// Dynamic Environment
class Actor{
    public:
    Object* obj;
    Vector2 velocity;
    bool is_grounded;

    void move(const float in_x, const float in_y, const float delta_time_ms){
        // Accelerate left and right
        velocity.x += in_x*env_constants.move_acc*delta_time_ms;
        // Jump with instantaneous speed
        if(in_y > 0  &&  is_grounded)
            velocity.y -= env_constants.jump_speed;
    }
};
class Enemy : public Actor{
    public:
    bool player_death_flag;
    bool fall_death_flag;
    bool player_kill_flag;
    std::vector<float> prev_choices;

    std::vector<float> prev_noises;
    std::vector<float> prev_smoothed_outputs;

    Episode* episode_history;

    bool isAboveTerrain() const{
        const int mid = AI_MAP_VISIBILITY/2; // this is only correct if AI_MAP_VISIBILITY is odd
        const float c_raycast = episode_history->getTimeStates().back().state_v[7 + mid];
        return c_raycast >= 0;
    }
};
struct EnvironmentDynamics{
    Actor player;
    int action_repeat_ticks;
    std::vector<Enemy> enemies;
    std::vector<Episode> enemy_episodes;
};

// Dynamic Environment Variables
bool resetFlag = false;
EnvironmentDynamics env_dynamics;

PPO ppo;


// Initialization
    // Request a new environment from the js.
    // In the future, we could potentially load all the environments we need at the beginning, instead of just one at a time.
    void requestNewEnvironment(){
        #ifdef EMSCRIPTEN
            // Request JS to pass a random environment
            EM_ASM(
                passRandomEnvironment();
            );
        #else
            std::cerr << "Requesting a new environment is not defined without Emscripten\n";
        #endif
    }
    // Randomize enemy and player positions on the terrain
    void randomizeActorPositions(){
        const int terrain_count = active_environment.terrain_count;
        const int actor_count = active_environment.enemy_count + 1;
        if(terrain_count == 0)
            return;

        // Create indexes to sample from
        std::vector<int> indexes(terrain_count);
        for(int i = 0; i < terrain_count; ++i)
            indexes[i] = i;

        // Choose spot for player
        const int p_terrain_index = (int)(randomRange()*terrain_count);
        indexes.erase(indexes.begin()+p_terrain_index);
        // Set spot on top of terrain
        Object& p = active_environment.player;
        const Object& p_terrain = active_environment.terrain[p_terrain_index];
        const float p_x = p_terrain.pos.x + randomRange()*(p_terrain.size.x - p.size.x);
        const float p_y = p_terrain.pos.y - p.size.y - _EPSILON;
        p.pos = { p_x, p_y };

        // Choose spots for enemies
        if(indexes.size() == 0) indexes.push_back(0);
        for(int i = 0; i < active_environment.enemy_count  &&  indexes.size() > 0; i++){
            // Choose spot for player
            const int index = (int)(randomRange()*indexes.size());
            const int e_terrain_index = indexes[index];
            indexes.erase(indexes.begin()+index);
            // Set spot on top of terrain
            Object& e = active_environment.enemies[i];
            const Object& e_terrain = active_environment.terrain[e_terrain_index];
            const float e_x = e_terrain.pos.x + randomRange()*(e_terrain.size.x - e.size.x);
            const float e_y = e_terrain.pos.y - e.size.y - _EPSILON;
            e.pos = { e_x, e_y };
        }
    }

    // Resets all dynamic parts of the environment
    void resetEnvironment(){
        // Add summed rewards from the end of the round
        game_metadata.last_round_end_reward = 0;
        for(int i = 0; i < env_dynamics.enemy_episodes.size(); ++i){
            const auto& timestates = env_dynamics.enemy_episodes[i].getTimeStates();
            for(int j = 0; j < timestates.size(); ++j){
                game_metadata.last_round_end_reward += timestates[j].reward;
            }
        }

        // Check if env has been reused too long
        game_metadata.total_env_time_ms += game_metadata.env_time_ms;
        game_metadata.env_time_ms = 0;
        if(game_metadata.total_env_time_ms > MAX_TIME_IN_ENV_MS){
            game_metadata.total_env_time_ms = 0;
            requestNewEnvironment();
        }

        // Initialize environment
        active_environment = initial_environment;

        // Initialize dynamic player variables
        env_dynamics.player.obj = &active_environment.player;
        env_dynamics.player.velocity = {0,0};
        env_dynamics.player.is_grounded = false;

        // Initialize repeat time
        env_dynamics.action_repeat_ticks = 0;

        // Initialize dynamic enemy variables 
        env_dynamics.enemies = std::vector<Enemy>(active_environment.enemy_count);
        env_dynamics.enemy_episodes = std::vector<Episode>(active_environment.enemy_count);
        for(int i = 0; i < active_environment.enemy_count; i++){
            // Set enemy variables to default
            env_dynamics.enemies[i].obj = &active_environment.enemies[i];
            env_dynamics.enemies[i].velocity = {0,0};
            env_dynamics.enemies[i].is_grounded = false;
            env_dynamics.enemies[i].player_death_flag = false;
            env_dynamics.enemies[i].fall_death_flag = false;
            env_dynamics.enemies[i].player_kill_flag = false;
            env_dynamics.enemies[i].prev_noises = std::vector<float>(POLICY_OUTPUT_LAYER_SIZE/2, 0);
            env_dynamics.enemies[i].prev_smoothed_outputs = std::vector<float>(POLICY_OUTPUT_LAYER_SIZE, 0);
            env_dynamics.enemies[i].prev_choices = std::vector<float>(POLICY_OUTPUT_LAYER_SIZE/2, 0);

            // Link episode to enemy
            env_dynamics.enemies[i].episode_history = &env_dynamics.enemy_episodes[i];
        }
        
        // Randomize positions of enemies and the player
        randomizeActorPositions();

        // Reset flag
        resetFlag = false;
    }

// External Initialization
    // Initializes the environment to have objects set in JS
    // Returns the pointer to the initial environment
    extern "C" EMSCRIPTEN_KEEPALIVE Environment* initializeEnvironment(){
        return &initial_environment;
    }
    
    extern "C" EMSCRIPTEN_KEEPALIVE void setEnvironmentConstants(   const float gravity, const float drag, 
                                                                    const float move_acc, const float jump_speed,
                                                                    const float enemy_visibility){
        // Initialize environment constants
        env_constants.gravity = gravity;
        env_constants.drag = drag;
        env_constants.move_acc = move_acc;
        env_constants.jump_speed = jump_speed;
        env_constants.enemy_visibility = enemy_visibility;

        // Compute other constants
        env_constants.max_move_speed = (move_acc/drag);
        env_constants.max_fall_speed = (gravity/drag);
        // Approximate max jump height
        const float delta_time_ms = 33.333f;
        float v = -jump_speed;
        float h = 0;
        while(v < 0){
            v += gravity * delta_time_ms;
            v -= v * drag * delta_time_ms;
            h -= v * delta_time_ms;
        }
        env_constants.max_jump_height = h;
    }

    // Sets any neccessary variables the environment has been set in JS
    // Returns pointer to start of the active environment memory
    extern "C" EMSCRIPTEN_KEEPALIVE Environment* finalizeEnvironment(){
        // Compute bounds
        env_constants.min_bound = {_INFINITY, _INFINITY}; 
        env_constants.max_bound = {-_INFINITY, -_INFINITY}; 
        for(int i = 0; i < initial_environment.terrain_count; ++i){
            if(initial_environment.terrain[i].pos.y + initial_environment.terrain[i].size.y > env_constants.max_bound.y)   env_constants.max_bound.y = initial_environment.terrain[i].pos.y + initial_environment.terrain[i].size.y;
            if(initial_environment.terrain[i].pos.y < env_constants.min_bound.y)                                          env_constants.min_bound.y = initial_environment.terrain[i].pos.y;
            if(initial_environment.terrain[i].pos.x + initial_environment.terrain[i].size.x > env_constants.max_bound.x)   env_constants.max_bound.x = initial_environment.terrain[i].pos.x + initial_environment.terrain[i].size.x;
            if(initial_environment.terrain[i].pos.x < env_constants.min_bound.x)                                          env_constants.min_bound.x = initial_environment.terrain[i].pos.x;
        }
        env_constants.min_bound.y -= BOUNDS_BUFFER_MULTIPLIER * initial_environment.player.size.y; 
        env_constants.max_bound.y += BOUNDS_BUFFER_MULTIPLIER * initial_environment.player.size.y;
        env_constants.min_bound.x -= BOUNDS_BUFFER_MULTIPLIER * initial_environment.player.size.x; 
        env_constants.max_bound.x += BOUNDS_BUFFER_MULTIPLIER * initial_environment.player.size.x;

        return &active_environment;
    }

    // Initializes the neural networks with the given sizes to be ready to have weights set in JS
    // Returns the pointer to the network info
    extern "C" EMSCRIPTEN_KEEPALIVE NetworkPointers* initializeNetwork( const int value_hidden_layer_size, const int value_hidden_layer_count,
                                                                        const int policy_hidden_layer_size, const int policy_hidden_layer_count){
        // Create PPO with given sizes
        ppo = PPO(  INPUT_LAYER_SIZE, POLICY_OUTPUT_LAYER_SIZE,
                    value_hidden_layer_size, value_hidden_layer_count,
                    policy_hidden_layer_size, policy_hidden_layer_count);
        
        // Add beginning of each weight array to the network_ptrs
        network_ptrs.value_hidden_layer_size = value_hidden_layer_size;
        network_ptrs.value_hidden_layer_count = value_hidden_layer_count;
        network_ptrs.policy_hidden_layer_size = policy_hidden_layer_size;
        network_ptrs.policy_hidden_layer_count = policy_hidden_layer_count;
        network_ptrs.value_weights = ppo.value_network.getWeightLayer(0);
        network_ptrs.value_bias = ppo.value_network.getBiasLayer(0);
        network_ptrs.policy_weights = ppo.policy_network.getWeightLayer(0);
        network_ptrs.policy_bias = ppo.policy_network.getBiasLayer(0);

        return &network_ptrs;
    }
    // Initializes the neural networks with random weights, instead of setting them in JS
    // Returns the pointer to the network info
    extern "C" EMSCRIPTEN_KEEPALIVE NetworkPointers* initializeRandomNetwork(){
        // Create network with default sizes
        initializeNetwork(VALUE_DEFAULT_HIDDEN_LAYER_SIZE, VALUE_DEFAULT_HIDDEN_LAYER_COUNT, POLICY_DEFAULT_HIDDEN_LAYER_SIZE, POLICY_DEFAULT_HIDDEN_LAYER_COUNT);
        
        // Randomize weights
        ppo.initialize();

        return &network_ptrs;
    }
    // Sets any neccessary variables after weights have been set in JS
    extern "C" EMSCRIPTEN_KEEPALIVE void finalizeNetwork(){
        // Reset episode, if network was set in the middle of an episode
        env_dynamics.enemy_episodes = std::vector<Episode>(active_environment.enemy_count);
        for(int i = 0; i < active_environment.enemy_count; i++){
            // Link episode to enemy
            env_dynamics.enemies[i].episode_history = &env_dynamics.enemy_episodes[i];
        }
    }

    // Finalizes everything else needed to start and do updates
    // Called once at start
    extern "C" EMSCRIPTEN_KEEPALIVE GameMetadata* finalizeGame(){
        srand(time(0));

        // Set external facing metadata
        game_metadata.player_wins = 0;
        game_metadata.player_losses = 0;
        for(int i = 0; i < MAX_ENEMY_COUNT; ++i){
            game_metadata.enemy_rewards[i] = -1;
        }

        // Set environment times to 0
        game_metadata.env_time_ms = 0;
        game_metadata.total_env_time_ms = 0;
        game_metadata.total_time_ms = 0;

        // Initialize dynamic variables
        resetEnvironment();

        return &game_metadata;
    }


// Physics
    // Checks if two boxes are intersecting
    bool isIntersecting(const Vector2& pos_A, const Vector2& size_A, const Vector2& pos_B, const Vector2& size_B){
        return  pos_A.y < pos_B.y + size_B.y  &&  pos_A.y + size_A.y > pos_B.y  &&  // above && below &&
                pos_A.x < pos_B.x + size_B.x  &&  pos_A.x + size_A.x > pos_B.x;     // left && right
    }
    // Handles collisions between static terrain with player and enemies
    void doStaticCollisions(const float delta_time_ms){
        // First reset on ground flag
        env_dynamics.player.is_grounded = false;
        // Check collision between player and terrain
        const Vector2& p_pos = env_dynamics.player.obj->pos;
        const Vector2& p_size = env_dynamics.player.obj->size;
        for(int j = 0; j < active_environment.terrain_count; j++){
            const Vector2& t_pos = active_environment.terrain[j].pos;
            const Vector2& t_size = active_environment.terrain[j].size;
            // Check intersection
            if( isIntersecting(p_pos, p_size, t_pos, t_size) ){
                const Vector2 delta_pos = {env_dynamics.player.velocity.x * delta_time_ms, env_dynamics.player.velocity.y * delta_time_ms};
                // Move to previous position to determine collision direction
                if( p_pos.y - delta_pos.y > t_pos.y + t_size.y ){ // above
                    env_dynamics.player.velocity.y = 0;
                    env_dynamics.player.obj->pos.y = t_pos.y + t_size.y + _EPSILON;
                }
                else if( p_pos.y + p_size.y - delta_pos.y < t_pos.y ){ //below
                    env_dynamics.player.velocity.y = 0;
                    env_dynamics.player.obj->pos.y = t_pos.y - p_size.y - _EPSILON;
                    env_dynamics.player.is_grounded = true;
                }
                if( p_pos.x - delta_pos.x > t_pos.x + t_size.x ){ //left
                    env_dynamics.player.velocity.x = 0;
                    env_dynamics.player.obj->pos.x = t_pos.x + t_size.x + _EPSILON;
                }
                else if( p_pos.x + p_size.x - delta_pos.x < t_pos.x ){ //right
                    env_dynamics.player.velocity.x = 0;
                    env_dynamics.player.obj->pos.x = t_pos.x - p_size.x - _EPSILON;
                }
            }
        }

        // Check collision between each enemy and terrain (same as player collisions)
        // Potentially merge player with enemy array, so player and enemy collisions aren't separate
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            // First reset on ground flag
            env_dynamics.enemies[i].is_grounded = false;
            // Check collision between player and terrain
            const Vector2& e_pos = env_dynamics.enemies[i].obj->pos;
            const Vector2& e_size = env_dynamics.enemies[i].obj->size;
            for(int j = 0; j < active_environment.terrain_count; j++){
                const Vector2& t_pos = active_environment.terrain[j].pos;
                const Vector2& t_size = active_environment.terrain[j].size;
                // Check intersection
                // above, below, left, right
                if( isIntersecting(e_pos, e_size, t_pos, t_size) ){
                    const Vector2 delta_pos = {env_dynamics.enemies[i].velocity.x * delta_time_ms, env_dynamics.enemies[i].velocity.y * delta_time_ms};
                    // Move to previous position to determine collision direction
                    if( e_pos.y - delta_pos.y > t_pos.y + t_size.y ){ // above
                        env_dynamics.enemies[i].velocity.y = 0;
                        env_dynamics.enemies[i].obj->pos.y = t_pos.y + t_size.y + _EPSILON;
                    }
                    else if( e_pos.y + e_size.y - delta_pos.y < t_pos.y ){ //below
                        env_dynamics.enemies[i].velocity.y = 0;
                        env_dynamics.enemies[i].obj->pos.y = t_pos.y - e_size.y - _EPSILON;
                        env_dynamics.enemies[i].is_grounded = true;
                    }
                    if( e_pos.x - delta_pos.x > t_pos.x + t_size.x ){ // left
                        env_dynamics.enemies[i].velocity.x = 0;
                        env_dynamics.enemies[i].obj->pos.x = t_pos.x + t_size.x + _EPSILON;
                    }
                    else if( e_pos.x + e_size.x - delta_pos.x < t_pos.x ){ //right
                        env_dynamics.enemies[i].velocity.x = 0;
                        env_dynamics.enemies[i].obj->pos.x = t_pos.x - p_size.x - _EPSILON;
                    }
                }
            }
        }
    }
    // All non-input physics updates
    void doPhysicsUpdate(const float delta_time_ms){
        // Update velocity with gravity
        const float g_velocity = env_constants.gravity * delta_time_ms;
        env_dynamics.player.velocity.y += g_velocity;
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            env_dynamics.enemies[i].velocity.y += g_velocity;
        }

        // Apply drag based on speed
        const float drag_c = env_constants.drag * delta_time_ms;
        env_dynamics.player.velocity.x -= env_dynamics.player.velocity.x * drag_c;
        env_dynamics.player.velocity.y -= env_dynamics.player.velocity.y * drag_c;
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            env_dynamics.enemies[i].velocity.x -= env_dynamics.enemies[i].velocity.x * drag_c;
            env_dynamics.enemies[i].velocity.y -= env_dynamics.enemies[i].velocity.y * drag_c;
        }

        // Update positions of player and enemies based on their velocities
        env_dynamics.player.obj->pos.x += env_dynamics.player.velocity.x * delta_time_ms;
        env_dynamics.player.obj->pos.y += env_dynamics.player.velocity.y * delta_time_ms;
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            env_dynamics.enemies[i].obj->pos.x += env_dynamics.enemies[i].velocity.x * delta_time_ms;
            env_dynamics.enemies[i].obj->pos.y += env_dynamics.enemies[i].velocity.y * delta_time_ms;
        }

        // Do collisions
        doStaticCollisions(delta_time_ms);
    }


// Enemy AI
    // Finds the relative heightmap around the enemy with radius of the enemy view radius, and with AI_MAP_VISIBILITY heights
    // Heightmap points are spaced evenly with two points always being on the edges.
    void relativeHeightmap(const Vector2 &c_pos, float* heightmap){
        // Initialize all values at infinity
        for(int i = 0; i < AI_MAP_VISIBILITY; ++i)
            heightmap[i] = _INFINITY;
        
        const float v_x_min = c_pos.x - env_constants.enemy_visibility;
        const float v_x_max = c_pos.x + env_constants.enemy_visibility;
        const float spacing = 2*env_constants.enemy_visibility / (AI_MAP_VISIBILITY-1);
        // Only count terrain below the enemies jump height
        const float raycast_y = c_pos.y - env_constants.max_jump_height * 1.1f;
        const float max_raycast_y = c_pos.y + env_constants.enemy_visibility*VERTICAL_VISIBILITY_MULTIPLIER;

        // Iterate through each piece of static terrain
        for(int j = 0; j < active_environment.terrain_count; j++){
            const Object& terrain = active_environment.terrain[j];
            const float t_x_min = terrain.pos.x;
            const float t_x_max = terrain.pos.x + terrain.size.x;
            const float t_surface_y = terrain.pos.y;

            // Check if terrain in the bounds
            if(t_x_max < v_x_min  ||  v_x_max < t_x_min)
                continue;
            // Check if terrain is at the correct height (not above jumping height) and not too far below
            if(t_surface_y < raycast_y  ||  max_raycast_y < t_surface_y)
                continue;
            // If terrain is a roof overhead, don't use it, even if it is technically lower than jump height
            if(t_x_min < c_pos.x &&  c_pos.x < t_x_max  &&  t_surface_y < c_pos.y)
                continue;

            // Iterate through each x value for the heightmap
            float h_x = v_x_min - spacing;
            for(int i = 0; i < AI_MAP_VISIBILITY; ++i){
                h_x += spacing;

                // Check if point is within x bounds of terrain
                if(h_x < t_x_min)
                    continue;
                else if(t_x_max < h_x)
                    break;
                
                // Check if current relative height is smaller than previously set
                const float rel_h = t_surface_y - raycast_y;
                if( rel_h < heightmap[i])
                    heightmap[i] = rel_h;
            }
        }

        // Set infinity values to -1 and normalize the other values
        float max_ray_distance = max_raycast_y - raycast_y;
        for(int i = 0; i < AI_MAP_VISIBILITY; ++i){
            if(heightmap[i] == _INFINITY){   
                heightmap[i] = -1;
            }
            else{
                heightmap[i] /= max_ray_distance;
                // Invert heightmap so the highest y is 1, and the lowest is 0
                heightmap[i] = 1 - heightmap[i];
            }
        }
    }

    // Computes the reward for an enemy in a given state
    float rewardFunction(const Enemy& enemyData){
        float reward = 0;

        // Punish the enemy for being killed
        if(enemyData.fall_death_flag){
            reward += FALL_DEATH_REWARD;
        }
        else if(enemyData.player_death_flag){
            reward += PLAYER_DEATH_REWARD;
        }
        else{
            // Reward the enemy for being alive for longer
            reward += ALIVE_REWARD;

            // Punish the enemy for falling and not having anything beneath them (falling off edge)
            if(enemyData.velocity.y > _EPSILON  &&  !enemyData.isAboveTerrain())
                reward +=  FALLING_REWARD * (enemyData.velocity.y / env_constants.max_fall_speed);

            // Reward the enemy for moving as a fraction of its max speed (max speed is equal to move_acc/drag)
            reward += MOVE_REWARD * (std::abs(enemyData.velocity.x) / env_constants.max_move_speed);

            // Punish for being too far away
            const float x = enemyData.episode_history->getTimeStates().back().getStateData()[0];
            const float y = enemyData.episode_history->getTimeStates().back().getStateData()[1];
            const float dist = std::sqrt(x*x + y*y);
            reward += DISTANCE_REWARD * dist;

            // Reward the enemy for killing the player
            if(enemyData.player_kill_flag){
                reward += PLAYER_KILL_REWARD;
            }
        }
        return reward;
    }
    // Updates each enemies velocity
    void doEnemyUpdate(const float delta_time_ms){
        // If repeating actions, move enemies with their saved action
        if(env_dynamics.action_repeat_ticks > 0){
            for(int i = 0; i < env_dynamics.enemies.size(); ++i){
                const float* aug_choices = env_dynamics.enemies[i].prev_choices.data();
                env_dynamics.enemies[i].move(aug_choices[0], aug_choices[1], delta_time_ms);
            }
            return;
        }

        // Create random number generator
        std::random_device rd;
        std::mt19937 rd_gen(rd());

        // Store player position
        const Vector2 p_c_pos = {env_dynamics.player.obj->pos.x + env_dynamics.player.obj->size.x/2, env_dynamics.player.obj->pos.y + env_dynamics.player.obj->size.y/2};;
        const Vector2& p_v = env_dynamics.player.velocity;
        // Move each enemy
        for(int i = 0; i < env_dynamics.enemies.size(); ++i){
            // Get enemy's center position
            const Vector2 e_c_pos = {env_dynamics.enemies[i].obj->pos.x + env_dynamics.enemies[i].obj->size.x/2, env_dynamics.enemies[i].obj->pos.y + env_dynamics.enemies[i].obj->size.y/2};
            
            // Store inputs and outputs in the enemies current timestate
            TimeState& this_timestate = env_dynamics.enemies[i].episode_history->addModifableTimeState(delta_time_ms);

            // Get relevant inputs
            float* in = this_timestate.getStateData();
                // Player distance and velocity
                const float y_diff = p_c_pos.y - e_c_pos.y;
                const float x_diff = p_c_pos.x - e_c_pos.x;
                
                #if defined( MANUAL_TRAIN )
                const float dist = std::sqrt(y_diff*y_diff + x_diff*x_diff);
                if( dist < env_constants.enemy_visibility ){ // Only pass info if the player is within range
                    in[0] = x_diff / env_constants.enemy_visibility; 
                    in[1] = y_diff / env_constants.enemy_visibility;
                    in[2] = p_v.x  / env_constants.max_move_speed;
                    in[3] = p_v.y  / env_constants.jump_speed;
                }
                else{
                    in[0] = 0; in[1] = 0; 
                    in[2] = 0; in[3] = 0;
                }
                #elif defined( TARGET_TRAIN )
                    const float env_width = env_constants.max_bound.x - env_constants.min_bound.x;
                    //const float env_height = env_constants.max_bound.y - env_constants.min_bound.y;
                    in[0] = x_diff / env_width;
                    in[1] = y_diff / env_width;

                    in[2] = p_v.x  / env_constants.max_move_speed; 
                    in[3] = p_v.y  / env_constants.jump_speed;
                #elif defined( MOVE_TRAIN )
                    in[0] = 0; in[1] = 0;
                    in[2] = 0; in[3] = 0;
                #endif
                
                // Enemy velocity
                in[4] = env_dynamics.enemies[i].velocity.x / env_constants.max_move_speed;
                in[5] = env_dynamics.enemies[i].velocity.y / env_constants.jump_speed;
                in[6] = env_dynamics.enemies[i].is_grounded;
                // Local heightmap around enemy
                relativeHeightmap(e_c_pos, (in+7));
            
            // Pass inputs to neural neuralNet and get outputs
            float* out = this_timestate.getActionData();
            ppo.policy_network.propagate(in, out);
            
            // Choose random value according to the outputted mean and log standard deviation
            float* prev_noises = env_dynamics.enemies[i].prev_noises.data(); 
            float* prev_smoothed_outputs = env_dynamics.enemies[i].prev_smoothed_outputs.data(); 
            float* aug_choices = env_dynamics.enemies[i].prev_choices.data(); 
            ppo.sampleChoice(out, prev_smoothed_outputs, prev_noises, this_timestate.getChoicesData(), aug_choices, rd_gen, -1, 1);

            // Use outputs to move enemy
            if(std::isnan(aug_choices[0])) aug_choices[0] = 0;
            if(std::isnan(aug_choices[1])) aug_choices[1] = 0;
            env_dynamics.enemies[i].move(aug_choices[0], aug_choices[1], delta_time_ms);

            //if(((double)rand()/RAND_MAX) < 0.05f){
            //    PRINT("START");
            //    PRINT_VAR(env_dynamics.enemies[i].prev_smoothed_outputs[2]);
            //    PRINT_VAR(env_dynamics.enemies[i].prev_smoothed_outputs[3]);
            //}
        }
    }
    // Computes the reward for enemies, and trains the AI if a reset will happen
    void doAIUpdate(const float delta_time_ms){
        // Update tick count
        env_dynamics.action_repeat_ticks++;
        if(env_dynamics.action_repeat_ticks >= ACTION_REPEAT_TICKS)
            env_dynamics.action_repeat_ticks = 0;

        bool player_killed = false;
        // Compute rewards for each enemy
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            
            // Get enemy reward based on it's current state
            const float reward = rewardFunction(env_dynamics.enemies[i]);
            game_metadata.enemy_rewards[i] = reward;

            // Save enemy's reward in it's episode
            env_dynamics.enemies[i].episode_history->addRewardToBack(reward);

            if(env_dynamics.enemies[i].fall_death_flag  ||  env_dynamics.enemies[i].player_death_flag){
                // Then delete the enemy so no more data is collected about it
                active_environment.enemies[i] = active_environment.enemies[active_environment.enemy_count-1];
                active_environment.enemy_count--;
                env_dynamics.enemies[i] = env_dynamics.enemies.back();
                env_dynamics.enemies.pop_back();
                env_dynamics.enemies[i].obj = &active_environment.enemies[i];
                --i;

                // Trigger a reset if all enemies are gone
                if(env_dynamics.enemies.size() == 0){
                    resetFlag = true;
                    game_metadata.player_wins ++;
                    break;
                }
            }
            // If the player was killed, only train the enemy that killed the player
            else if(env_dynamics.enemies[i].player_kill_flag){
                ppo.train( *env_dynamics.enemies[i].episode_history );
                player_killed = true;
                break;
            }
        }

        // If we are resetting, train the policy on each enemy's trajectory
        if(resetFlag  &&  !player_killed){
            ppo.train(env_dynamics.enemy_episodes);
        }
    }

// Game Managment
    void playerDeath(){
        resetFlag = true;
        game_metadata.player_losses ++;
    }
    // Does miscellaneous game updates, like player and enemy deaths
    void doGameUpdate(const float delta_time_ms){
        // Check if player is out of bounds
        const Vector2& p_pos = env_dynamics.player.obj->pos;
        const Vector2& p_size = env_dynamics.player.obj->size;
        if( p_pos.y + p_size.y < env_constants.min_bound.y  ||  p_pos.y > env_constants.max_bound.y  ||   //above, below
            p_pos.y + p_size.x < env_constants.min_bound.x  ||  p_pos.x > env_constants.max_bound.x){     //left, right
            // Player death to falling
            playerDeath();
            return;
        }

        // Check if enemies are out of bounds
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            const Vector2& e_pos = env_dynamics.enemies[i].obj->pos;
            const Vector2& e_size = env_dynamics.enemies[i].obj->size;
            if( e_pos.y + e_size.y < env_constants.min_bound.y  ||  e_pos.y > env_constants.max_bound.y  ||   //above, below
                e_pos.y + e_size.x < env_constants.min_bound.x  ||  e_pos.x > env_constants.max_bound.x){     //left, right
                // Enemy death to falling
                env_dynamics.enemies[i].fall_death_flag = true;
            }
        }

        // Don't do player collisions if we are training movement
        #if defined( MOVE_TRAIN )
        return;
        #endif

        // Check collision with player and enemies
        for(int i = 0; i < env_dynamics.enemies.size(); i++){
            const Vector2& e_pos = env_dynamics.enemies[i].obj->pos;
            const Vector2& e_size = env_dynamics.enemies[i].obj->size;
            // Check intersection
            if( isIntersecting(p_pos, p_size, e_pos, e_size) ){
                const Vector2 p_delta_pos = {env_dynamics.player.velocity.x * delta_time_ms, env_dynamics.player.velocity.y * delta_time_ms};
                const Vector2 e_delta_pos = {env_dynamics.enemies[i].velocity.x * delta_time_ms, env_dynamics.enemies[i].velocity.y * delta_time_ms};
                // Move to previous position to determine collision direction
                if( p_pos.y - p_delta_pos.y > e_pos.y + e_size.y - e_delta_pos.y){ // above: player dies
                    // Player death to enemy
                    playerDeath();
                    env_dynamics.enemies[i].player_kill_flag = true;
                    return;
                }
                else if( p_pos.y + p_size.y - p_delta_pos.y < e_pos.y - e_delta_pos.y ){ //below: enemy dies
                    // Enemy death to player
                    env_dynamics.enemies[i].player_death_flag = true;
                    continue;
                }
                if( p_pos.x - p_delta_pos.x > e_pos.x + e_size.x - e_delta_pos.x ){ //left: player dies
                    // Player death to enemy
                    playerDeath();
                    env_dynamics.enemies[i].player_kill_flag = true;
                    return;
                }
                else if( p_pos.x + p_size.x - p_delta_pos.x < e_pos.x - e_delta_pos.x ){ //right: player dies
                    // Player death to enemy
                    playerDeath();
                    env_dynamics.enemies[i].player_kill_flag = true;
                    return;
                }
            }
        }
    }


// Updates
    // Updates player velocity from the inputs
    void doPlayerUpdate(const float delta_time_ms, const float player_in_x, const float player_in_y){
        env_dynamics.player.move(player_in_x, player_in_y, delta_time_ms);
    }
    
    // Update the simulation by one tick
    extern "C" EMSCRIPTEN_KEEPALIVE void doUpdate(const float delta_time_ms, const float player_in_x, const float player_in_y, const int speedup = 1){
        for(int i = 0; i < speedup; ++i){
            // Reset environment if anything called for it
            if(resetFlag)
                resetEnvironment();
            if( game_metadata.env_time_ms > MAX_SIMULATION_TIME )
                resetFlag = true;
            
            // Player movement
            doPlayerUpdate(delta_time_ms, player_in_x, player_in_y);
            // Enemy movement
            doEnemyUpdate(delta_time_ms);
            // Physics (gravity, collision, drag)
            doPhysicsUpdate(delta_time_ms);
            // Player/enemy death
            doGameUpdate(delta_time_ms);
            // Compute AI rewards and train AI
            doAIUpdate(delta_time_ms);

            // Update time
            game_metadata.env_time_ms += delta_time_ms;
            game_metadata.total_time_ms += delta_time_ms;
        }
    }



/*
class LossyMap{
    public:
    LossyMap(){}
    LossyMap(const Environment& init_env, const EnvironmentConstants& env_c){
        // Determine map size
        min_env_bound = env_c.min_bound;
        map_res = env_c.enemy_visibility*1.41421356237 / AI_MAP_VISIBILITY;
        const int map_height = (env_c.max_bound.y - min_env_bound.y - 2*BOUNDS_BUFFER) / map_res;
        const int map_width = (env_c.max_bound.x - min_env_bound.x - 2*BOUNDS_BUFFER) / map_res;
        // Initialize map
        map = std::vector<std::vector<float>>(map_height);
        for(int i = 0; i < map_height; i++)
            map[i] = std::vector<float>(map_width, NO_TERRAIN);

        // Populate map with terrain
        for(int k = 0; k < init_env.terrain_count; k++){
            const Vector2& t_pos = init_env.terrain[k].pos;
            const Vector2& t_size = init_env.terrain[k].size;
            // Set indexes to 1 where the terrain is
            const int y1 = (t_pos.y - min_env_bound.y) / map_res;
            const int y2 = (t_pos.y + t_size.y - min_env_bound.y) / map_res;
            const int x1 = (t_pos.x - min_env_bound.x) / map_res;
            const int x2 = (t_pos.x + t_size.x - min_env_bound.x) / map_res;
            for(int i = y1; i < y2; i++){
                for(int j = x1; j < x2; j++){
                    map[i][j] = TERRAIN;
                }
            }
        }
    }
    
    void getLocalMap(const Vector2& c_pos, float* local_map) const{        
        const int c_y = (c_pos.y - min_env_bound.y) / map_res;
        const int c_x = (c_pos.x - min_env_bound.x) / map_res;
        const int y_start = c_y - AI_MAP_VISIBILITY/2;
        const int x_start = c_x - AI_MAP_VISIBILITY/2;
        // Set values centered on the center indexes
        for(int i = 0; i < AI_MAP_VISIBILITY; i++){
            const int y = y_start + i;
            if(y < 0  ||  y >= map.size()){
                for(int l = 0; l < map.size(); l++)
                    local_map[i*AI_MAP_VISIBILITY + l] = 0;
                continue;
            }

            for(int j = 0; j < AI_MAP_VISIBILITY; j++){
                const int x = x_start + j;
                if(x < 0  ||  x >= map[y].size()){
                    local_map[i*AI_MAP_VISIBILITY + j] = 0;
                    continue;
                }

                local_map[i*AI_MAP_VISIBILITY + j] = map[y][x];
            }
        }
    }

    private:
    float map_res;
    Vector2 min_env_bound;
    std::vector<std::vector<float>> map;
    constexpr float static NO_TERRAIN = 0;//0.25f;
    constexpr static float TERRAIN = 1;//0.75f;
};
*/