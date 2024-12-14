import Module from "/hivemind/wasm/wasm.js";

const ENVIRONMENT_COUNT = 3;

// WASM Module
var module;
var activeEnvPtr;
var networksPtr;
var metadataPtr;

// Game
var environments;
var speed_modifier = 1;
var tick_time;
var time_of_last_tick;
var screen_width;
var enemy_visibility;

// Canvas variables
var canvas = document.getElementById("game_canvas");
var canvas_context = canvas.getContext("2d");


// On window load
window.onload = async ()=>{
    // Initialize canvas with pixel size of browser window
    initializeCanvas();

    // Load WASM module
    module ??= await Module();

    // Get environments to use for training
    const set_name = new URL(window.location.href).searchParams.get("set");
    if(set_name == null){
        await setTrainingSet("set_2");
    }
    else{
        await setTrainingSet(set_name);
    }
    // Pass random environment from training set
    passRandomEnvironment();

    // Pass neural network weights to WASM
    const network_name = new URL(window.location.href).searchParams.get("network");
    if(network_name == null){
        //await setNetwork("base02_1-18_1-18");
        passNetwork(null);
    }
    else{
        await setNetwork(network_name);
    }

    // Finalize game
    metadataPtr = module._finalizeGame();

    // Call this to start the game loop
    startGameTicks();
};

// Set canvas to correct size
function initializeCanvas(){
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
}

// On window resize
window.addEventListener('resize', () => {
    // Resize canvas when screen is changed
    initializeCanvas();
    renderEnvironment();
});
// On window unload
window.onbeforeunload = function() {
    //return "Data will be lost if you leave the page, are you sure?";
};


// Get training constants and environments from server
async function getTrainingSet(set_name){
    const result = await fetch("/hivemind/data/training_sets/" + set_name +".json");
    const set = await result.json();
    return set;
}
async function cacheEnvironments(training_set){
    last_index = training_set.environments.length;

    environments = [];
    for(var i = 0; i < last_index; ++i){
        const env = await getEnvironment(training_set.environments[i]);
        environments.push(env);
    }
}
// Get training constants and environments from server and caches the environments
async function setTrainingSet(set_name){
    const set = await getTrainingSet(set_name);
    await cacheEnvironments(set);
}

// Get environment json from server
async function getEnvironment(env_name){
    const result = await fetch("/hivemind/data/environments/"+env_name+".json");
    const environment = await result.json();
    return environment;
}
var last_index = 0;
export function passRandomEnvironment(){
    // Get random env index
    var index = Math.floor(Math.random()*(environments.length-1));
    if(index >= last_index  &&  environments.length != 1)
        index++;
    last_index = index;

    // Get and pass environment
    passEnvironment(environments[index]);
}
// Start and pass environment to the WebAssembly
function passEnvironment(environment){
    // Set JS variables
    screen_width = environment.screen_width;
    tick_time = 1000.0/environment.tps;
    enemy_visibility = environment.enemy_visibility;

    // Create ptr
    const initialEnvPtr = module._initializeEnvironment();

    // Pass environment size to wasm
    module.HEAPU32[(initialEnvPtr)>>2] = environment.terrain.length;    //terrain_count
    module.HEAPU32[(initialEnvPtr+4)>>2] = environment.enemies.length;  //enemy_count

    // Pass player
    module.HEAPF32[(initialEnvPtr+8) >>2]  = (environment.player.x - environment.player.w/2)   //player.x
    module.HEAPF32[(initialEnvPtr+12) >>2] = (environment.player.y - environment.player.h/2)   //player.y
    module.HEAPF32[(initialEnvPtr+16) >>2] = environment.player.w   //player.w
    module.HEAPF32[(initialEnvPtr+20) >>2] = environment.player.h   //player.h

    // Pass terrain
    for(var i = 0; i < environment.terrain.length; i++){
        module.HEAPF32[(initialEnvPtr+24+(i*16)) >>2] = (environment.terrain[i].x - environment.terrain[i].w/2)    //terrain[i].x
        module.HEAPF32[(initialEnvPtr+28+(i*16)) >>2] = (environment.terrain[i].y - environment.terrain[i].h/2)    //terrain[i].y
        module.HEAPF32[(initialEnvPtr+32+(i*16)) >>2] = environment.terrain[i].w    //terrain[i].w
        module.HEAPF32[(initialEnvPtr+36+(i*16)) >>2] = environment.terrain[i].h    //terrain[i].h
    }

    // Pass enemies
    for(var i = 0; i < environment.enemies.length; i++){
        module.HEAPF32[(initialEnvPtr+1624+(i*16)) >>2] = (environment.enemies[i].x - environment.enemies[i].w/2)  //enemies[i].x
        module.HEAPF32[(initialEnvPtr+1628+(i*16)) >>2] = (environment.enemies[i].y - environment.enemies[i].h/2) //enemies[i].y
        module.HEAPF32[(initialEnvPtr+1632+(i*16)) >>2] = environment.enemies[i].w  //enemies[i].w
        module.HEAPF32[(initialEnvPtr+1636+(i*16)) >>2] = environment.enemies[i].h  //enemies[i].h
    }

    // Pass constants to environment
    module._setEnvironmentConstants(environment.gravity, 
                                    environment.drag, 
                                    environment.move_acc, 
                                    environment.jump_speed, 
                                    environment.enemy_visibility);

    // Finalize environment to be ready for running
    activeEnvPtr = module._finalizeEnvironment();
}

// Render all elements in the wasm environment
function renderEnvironment(){
    // Clear canvas
    const width = canvas.width;
    const height = canvas.height;
    canvas_context.clearRect(0, 0, width, height);

    // Scale all coordinates into canvas coordinates, based on screen_width in json
    const r = width / screen_width;

    // Get position to render all other objects around
    const canvas_offset_x = width/2 - r * (module.HEAPF32[(activeEnvPtr+8) >>2] + module.HEAPF32[(activeEnvPtr+16) >>2]/2);
    const canvas_offset_y = height/2 - r * (module.HEAPF32[(activeEnvPtr+12) >>2] + module.HEAPF32[(activeEnvPtr+20) >>2]/2);

    // Get environment size
    const terrain_count = module.HEAPU32[(activeEnvPtr)>>2];
    const enemy_count = module.HEAPU32[(activeEnvPtr+4)>>2];

    // Draw circles around enemies
    canvas_context.fillStyle = "rgb(245,245,245)";
    for(var i = 0; i < enemy_count; i++){
        const w = r * module.HEAPF32[(activeEnvPtr+1632+(i*16)) >>2]/2;
        const h = r * module.HEAPF32[(activeEnvPtr+1636+(i*16)) >>2]/2;

        canvas_context.beginPath();
        canvas_context.arc(
            r * module.HEAPF32[(activeEnvPtr+1624+(i*16)) >>2] + canvas_offset_x + w,  //enemies[i].x
            r * module.HEAPF32[(activeEnvPtr+1628+(i*16)) >>2] + canvas_offset_y + h,  //enemies[i].y
            r * enemy_visibility,
            0,
            2 * Math.PI
        );
        canvas_context.fill();
    }

    // Draw terrain
    canvas_context.fillStyle = "black";
    for(var i = 0; i < terrain_count; i++){
        canvas_context.fillRect(
            r * module.HEAPF32[(activeEnvPtr+24+(i*16)) >>2] + canvas_offset_x,    //terrain[i].x
            r * module.HEAPF32[(activeEnvPtr+28+(i*16)) >>2] + canvas_offset_y,    //terrain[i].y
            r * module.HEAPF32[(activeEnvPtr+32+(i*16)) >>2],                      //terrain[i].w
            r * module.HEAPF32[(activeEnvPtr+36+(i*16)) >>2]                       //terrain[i].h
        );
    }

    // Draw enemies
    canvas_context.fillStyle = "red";
    canvas_context.font = r*3 + "px Arial";
    for(var i = 0; i < enemy_count; i++){
        const x = r * module.HEAPF32[(activeEnvPtr+1624+(i*16)) >>2] + canvas_offset_x;
        const y = r * module.HEAPF32[(activeEnvPtr+1628+(i*16)) >>2] + canvas_offset_y;
        const w = r * module.HEAPF32[(activeEnvPtr+1632+(i*16)) >>2];
        const h = r * module.HEAPF32[(activeEnvPtr+1636+(i*16)) >>2];
        canvas_context.fillRect(
            x,  //enemies[i].x
            y,  //enemies[i].y
            w,  //enemies[i].w
            h   //enemies[i].h
        );

        const reward = module.HEAPF32[(metadataPtr + 24 + i*4) >>2] * 100;
        canvas_context.fillText(Math.round(reward), x + w, y - h/2);
    }

    // Draw player
    canvas_context.fillStyle = "blue";
    canvas_context.fillRect(
        r * module.HEAPF32[(activeEnvPtr+8) >>2] + canvas_offset_x,    //player.x
        r * module.HEAPF32[(activeEnvPtr+12) >>2] + canvas_offset_y,   //player.y
        r * module.HEAPF32[(activeEnvPtr+16) >>2],                     //player.w
        r * module.HEAPF32[(activeEnvPtr+20) >>2]                      //player.h
    );

    // Draw round text in upper left corner
    canvas_context.fillStyle = "blue";
    canvas_context.font = 30 + "px Arial";
    const player_wins = module.HEAPU32[(metadataPtr + 0) >>2];
    canvas_context.fillText(player_wins, 10, 30);

    canvas_context.fillStyle = "red";
    canvas_context.font = 30 + "px Arial";
    const player_losses = module.HEAPU32[(metadataPtr + 4) >>2];
    canvas_context.fillText(player_losses, 10, 60);

    canvas_context.fillStyle = "black";
    canvas_context.font = 30 + "px Arial";
    const enemy_rewards = module.HEAPF32[(metadataPtr + 20) >>2];
    canvas_context.fillText(enemy_rewards.toFixed(2), 10, 90);

    // Draw time text in lower right corner
    canvas_context.fillStyle = "black";
    canvas_context.font = 30 + "px Arial";
    const env_time = module.HEAPF32[(metadataPtr + 8) >>2]  / 1000;
    canvas_context.fillText( env_time.toFixed(2) , width - 50 - 16*Math.floor(env_time).toString().length, height-50);
    const total_time = module.HEAPF32[(metadataPtr + 16) >>2] / 1000;
    canvas_context.fillText( total_time.toFixed(2) , width - 50 - 16*Math.floor(total_time).toString().length, height-10);

    // Draw speedup in lower left corner
    if(speed_modifier > 1)
        canvas_context.fillText("x" + speed_modifier, 10, height-10);
}
// Render all elements in the environment json
function renderInitialEnvironment(environment){
    // Clear canvas
    const width = canvas.width;
    const height = canvas.height;
    canvas_context.clearRect(0, 0, width, height);

    // Size ratio of canvas
    const r = width / environment.screen_width;

    // Get position to render all other objects around
    const center_x = width/2;
    const center_y = height/2;
    const player_x = environment.player.x;
    const player_y = environment.player.y;

    // Draw terrain
    canvas_context.fillStyle = "black";
    environment.terrain.forEach(terrain => {
        const c_x = r*(terrain.x - player_x) + center_x;
        const c_y = r*(terrain.y - player_y) + center_y;
        canvas_context.fillRect(c_x - r*terrain.w/2, c_y - r*terrain.h/2, r*terrain.w, r*terrain.h);
    });

    // Draw enemies
    canvas_context.fillStyle = "red";
    environment.enemies.forEach(enemy => {
        const c_x = r*(enemy.x - player_x) + center_x;
        const c_y = r*(enemy.y - player_y) + center_y;
        canvas_context.fillRect(c_x - r*enemy.w/2, c_y - r*enemy.h/2, r*enemy.w, r*enemy.h);
    });

    // Draw player
    canvas_context.fillStyle = "blue";
    canvas_context.fillRect(center_x - r*environment.player.w/2, center_y - r*environment.player.h/2, r*environment.player.w, r*environment.player.h);
}

// Get network weights json from server
async function getNetwork(network_name){
    const result = await fetch("/hivemind/data/networks/" + network_name + ".json");
    const network = await result.json();
    return network;
}
// Start and pass network to WebAssembly
function passNetwork(network){
    // If network is null, initialize a random network
    if(network == null){
        networksPtr = module._initializeRandomNetwork();
        module._finalizeNetwork();
        return;
    }

    // Otherwise set the weights from the json
    networksPtr = module._initializeNetwork(network.value.hidden_layer_size, network.value.hidden_layer_count, 
                                            network.policy.hidden_layer_size, network.policy.hidden_layer_count);

    // Pointers to weights and bias' of networks
    const v_weights_ptr = module.HEAPU32[(networksPtr+28)>>2];
    const v_bias_ptr = module.HEAPU32[(networksPtr+32)>>2];
    const p_weights_ptr = module.HEAPU32[(networksPtr+36)>>2];
    const p_bias_ptr = module.HEAPU32[(networksPtr+40)>>2];

    // Set value weights into WebAssembly
    for(let i = 0; i < network.value.weights.length; i++){
        module.HEAPF32[(v_weights_ptr + i*4)>>2] = network.value.weights[i];
    }
    for(let i = 0; i < network.value.bias.length; i++){
        module.HEAPF32[(v_bias_ptr + i*4)>>2] = network.value.bias[i];
    }
    // Set policy weights into WebAssembly
    for(let i = 0; i < network.policy.weights.length; i++){
        module.HEAPF32[(p_weights_ptr + i*4)>>2] = network.policy.weights[i];
    }
    for(let i = 0; i < network.policy.bias.length; i++){
        module.HEAPF32[(p_bias_ptr + i*4)>>2] = network.policy.bias[i];
    }

    module._finalizeNetwork();
}
// Gets network weights json from server and passes it to WebAssembly
export async function setNetwork(network_name){
    const network = await getNetwork(network_name);
    passNetwork(network);
}
// Get weights of current value and policy network from WebAssembly
function retrieveNetwork(){
    // Json to store network info in
    var network = {
        "value":{
            "hidden_layer_size": 0, "hidden_layer_count": 0,
            "weights":[], "bias":[]
        },
        "policy":{
            "hidden_layer_size": 0, "hidden_layer_count": 0,
            "weights":[], "bias":[]
        }
    };

    // Get sizes of networks
    const input_size = module.HEAPU32[(networksPtr)>>2];
    const v_output_size = module.HEAPU32[(networksPtr+4)>>2];
    const p_output_size = module.HEAPU32[(networksPtr+8)>>2];

    const v_hidden_layer_size = module.HEAPU32[(networksPtr+12)>>2];
    const v_hidden_layer_count = module.HEAPU32[(networksPtr+16)>>2];
    const p_hidden_layer_size = module.HEAPU32[(networksPtr+20)>>2];
    const p_hidden_layer_count = module.HEAPU32[(networksPtr+24)>>2];

    // Set sizes into json
    network.value.hidden_layer_size = v_hidden_layer_size;
    network.value.hidden_layer_count = v_hidden_layer_count;
    network.policy.hidden_layer_size = p_hidden_layer_size;
    network.policy.hidden_layer_count = p_hidden_layer_count;

    // Compute number of weights and bias' of networks
    const v_weight_count = input_size*v_hidden_layer_size + v_hidden_layer_size*v_hidden_layer_size*(v_hidden_layer_count-1) + v_output_size*v_hidden_layer_size;
    const v_bias_count = v_hidden_layer_size*v_hidden_layer_count + v_output_size;
    const p_weight_count = input_size*p_hidden_layer_size + p_hidden_layer_size*p_hidden_layer_size*(p_hidden_layer_count-1) + p_output_size*p_hidden_layer_size;
    const p_bias_count = p_hidden_layer_size*p_hidden_layer_count + p_output_size;

    // Pointers to weights and bias' of networks
    const v_weights_ptr = module.HEAPU32[(networksPtr+28)>>2];
    const v_bias_ptr = module.HEAPU32[(networksPtr+32)>>2];
    const p_weights_ptr = module.HEAPU32[(networksPtr+36)>>2];
    const p_bias_ptr = module.HEAPU32[(networksPtr+40)>>2];

    // Set value weights into json
    for(let i = 0; i < v_weight_count; i++){
        network.value.weights.push( module.HEAPF32[(v_weights_ptr + i*4)>>2] );
    }
    for(let i = 0; i < v_bias_count; i++){
        network.value.bias.push( module.HEAPF32[(v_bias_ptr + i*4)>>2] );
    }
    // Set policy weights into json
    for(let i = 0; i < p_weight_count; i++){
        network.policy.weights.push( module.HEAPF32[(p_weights_ptr + i*4)>>2] );
    }
    for(let i = 0; i < p_bias_count; i++){
        network.policy.bias.push( module.HEAPF32[(p_bias_ptr + i*4)>>2] );
    }

    return network;
}


// Get what keys are pressed
let player_in_left = 0;
let player_in_right = 0;
let player_in_up = 0;
document.addEventListener('keydown', function(e) {
    if(e.key == 'a')
        player_in_left = 1;
    if(e.key == 'd')
        player_in_right = 1;
    if(e.key == 'w')
        player_in_up = 1;
});
document.addEventListener('keyup', function(e) {
    if(e.key == 'a')
        player_in_left = 0;
    if(e.key == 'd')
        player_in_right = 0;
    if(e.key == 'w')
        player_in_up = 0;
});

// Scroll wheel changes simulation speed
addEventListener("wheel", (e) => {
    speed_modifier -= e.deltaY/1;
    if(speed_modifier < 1)
        speed_modifier = 1;

    //startGameTicks();

    //console.log("Set speed modifier:", Math.floor(speed_modifier));
});

// Turn off simulation when not focused, so the delta_time doesn't go too high
let focused = true;
window.addEventListener('blur', ()=>{focused = false; player_in_left = player_in_right = player_in_up = 0;});
window.addEventListener('focus', ()=>{focused = true;});

// Starts game loop. Ticks according tick_time
var loop_interval;
function startGameTicks(){
    // Clear existing interval, if it exists
    if(loop_interval != null)
        clearInterval(loop_interval);

    // Initialize last tick time
    time_of_last_tick = performance.now();

    // Tick every 'tick_time' milliseconds
    loop_interval = setInterval(()=>{
        tick();
    }, tick_time);
}
var prev_network
// Ticks the simulation once
function tick(){
    const mod = Math.floor(speed_modifier);
    // Find delta_time
    const current_time = performance.now();
    const delta_time = mod==1 ? current_time - time_of_last_tick : tick_time;
    time_of_last_tick = current_time;
    
    // Skip timestep if deltatime is double the average delta_time_ms
    if(!focused  ||  delta_time > 2.0*tick_time)
        return;

    // Do tick update in WASM
    module._doUpdate(delta_time, (player_in_right-player_in_left), player_in_up, mod);
    
    // Render new update before next animation frame
    requestAnimationFrame(renderEnvironment);

    const network = retrieveNetwork();
    if(JSON.stringify(network) != JSON.stringify(prev_network)){
        console.log(network);
        prev_network = network;
    }
    
}