import Module from "/hivemind/wasm/wasm.js";
//import { renderEnvironment } from "../js/initializer";


// WASM Module
var module;
var activeEnvPtr;

// Game
const frameTime = 33.333;//30fps
var time_of_last_tick;

// Canvas variables
var canvas;
var canvas_context;
var screen_width;


// On thread load
self.addEventListener('message', async (e) => {
    console.log("Worker called!");

    // Set canvas
    canvas = e.data.canvas;
    canvas_context = canvas.getContext("2d");

    // Get Environment from JSON
    const environment = await getEnvironment();
    screen_width = environment.screen_width;

    // Load WASM module
    module ??= await Module();

    // Pass environment to WASM
    passEnvironment(environment);

    // This thread will be working on this loop for the rest of it's lifetime
    //module._startUpdates();
    startGameTicks();
});

// Get environment json from server
async function getEnvironment(){
    const result = await fetch("/hivemind/environments/env_1.json");
    const environment = await result.json();
    return environment;
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

// Start and pass environment to the Webassembly
async function passEnvironment(environment){
    // Create ptr
    var initialEnvPtr = module._getInitialPointer();

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

    // Pass initial to active
    activeEnvPtr = module._initializeEnvironment(environment.gravity);


    //module.HEAPF32[(initialEnvPtr+20)>>2] = 10;//player.h
    //module.HEAPU32[(initialEnvPtr)>>2] = 100; //terrain_count
    //module.HEAPF32[(initialEnvPtr+24)>>2] = 12; //terrain[0].x
}

// Render all elements in the wasm environment
function renderEnvironment(){
    // Clear canvas
    const width = canvas.width;
    const height = canvas.height;
    canvas_context.clearRect(0, 0, width, height);

    // Scale all coordinates into canvas coordinates, based on screen_width in json
    const r = canvas.width / screen_width;

    // Get position to render all other objects around
    const canvas_offset_x = width/2 - r * (module.HEAPF32[(activeEnvPtr+8) >>2] + module.HEAPF32[(activeEnvPtr+16) >>2]/2);
    const canvas_offset_y = height/2 - r * (module.HEAPF32[(activeEnvPtr+12) >>2] + module.HEAPF32[(activeEnvPtr+20) >>2]/2);

    // Get environment size
    const terrain_count = module.HEAPU32[(activeEnvPtr)>>2];
    const enemy_count = module.HEAPU32[(activeEnvPtr+4)>>2];

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
    for(var i = 0; i < enemy_count; i++){
        canvas_context.fillRect(
            r * module.HEAPF32[(activeEnvPtr+1624+(i*16)) >>2] + canvas_offset_x,  //enemies[i].x
            r * module.HEAPF32[(activeEnvPtr+1628+(i*16)) >>2] + canvas_offset_y,  //enemies[i].y
            r * module.HEAPF32[(activeEnvPtr+1632+(i*16)) >>2],                    //enemies[i].w
            r * module.HEAPF32[(activeEnvPtr+1636+(i*16)) >>2]                     //enemies[i].h
        );
    }

    // Draw player
    canvas_context.fillStyle = "blue";
    canvas_context.fillRect(
        r * module.HEAPF32[(activeEnvPtr+8) >>2] + canvas_offset_x,    //player.x
        r * module.HEAPF32[(activeEnvPtr+12) >>2] + canvas_offset_y,   //player.y
        r * module.HEAPF32[(activeEnvPtr+16) >>2],                     //player.w
        r * module.HEAPF32[(activeEnvPtr+20) >>2]                      //player.h
    );
}


function startGameTicks(){
    // Initialize last tick time
    time_of_last_tick = performance.now();

    // Tick every 'frameTime' milliseconds
    setInterval(()=>{
        requestAnimationFrame(tick);
    }, frameTime);
}

function tick(){
    // Find delta_time
    const current_time = performance.now();
    const delta_time = current_time - time_of_last_tick;
    time_of_last_tick = current_time

    // Do tick update in WASM
    module._doUpdate(delta_time);

    // Render new update
    renderEnvironment();
}