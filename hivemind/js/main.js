import {setNetwork} from "/hivemind/index.js"


// Set buttons
const info_popup = document.getElementById("info_popup");

const info_button = document.getElementById("info_button");
const close_button = document.getElementById("close_button");
info_button.addEventListener("click", (e)=>{
    openInfo();
});
close_button.addEventListener("click", (e)=>{
    closeInfo();
});
const network_buttons = document.getElementsByClassName("network_button");
for(var button of network_buttons){
    const network_name = button.id;
    
    button.addEventListener("click", (e)=>{
        console.log(network_name);
        let url = window.location.href;    
        let urlObj = new URL(url);
        
        var params = new URLSearchParams(urlObj);
        params.set("network", network_name);

        urlObj.search = params.toString();
        window.location.href = urlObj.toString();
    });
}



function openInfo(){
    info_popup.style.visibility = "visible";
}
function closeInfo(){
    info_popup.style.visibility = "hidden";
}

