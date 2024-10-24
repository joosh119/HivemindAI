
var Module = (() => {
  var _scriptName = import.meta.url;
  
  return (
async function(moduleArg = {}) {
  var moduleRtn;

var Module=moduleArg;var readyPromiseResolve,readyPromiseReject;var readyPromise=new Promise((resolve,reject)=>{readyPromiseResolve=resolve;readyPromiseReject=reject});var ENVIRONMENT_IS_WEB=typeof window=="object";var ENVIRONMENT_IS_WORKER=typeof importScripts=="function";var ENVIRONMENT_IS_NODE=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string"&&process.type!="renderer";if(ENVIRONMENT_IS_NODE){const{createRequire}=await import("module");let dirname=import.meta.url;if(dirname.startsWith("data:")){dirname="/"}var require=createRequire(dirname)}var moduleOverrides=Object.assign({},Module);var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");var nodePath=require("path");if(!import.meta.url.startsWith("data:")){scriptDirectory=nodePath.dirname(require("url").fileURLToPath(import.meta.url))+"/"}readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):nodePath.normalize(filename);var ret=fs.readFileSync(filename);return ret};readAsync=(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):nodePath.normalize(filename);return new Promise((resolve,reject)=>{fs.readFile(filename,binary?undefined:"utf8",(err,data)=>{if(err)reject(err);else resolve(binary?data.buffer:data)})})};if(!Module["thisProgram"]&&process.argv.length>1){thisProgram=process.argv[1].replace(/\\/g,"/")}arguments_=process.argv.slice(2);quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){if(ENVIRONMENT_IS_WORKER){scriptDirectory=self.location.href}else if(typeof document!="undefined"&&document.currentScript){scriptDirectory=document.currentScript.src}if(_scriptName){scriptDirectory=_scriptName}if(scriptDirectory.startsWith("blob:")){scriptDirectory=""}else{scriptDirectory=scriptDirectory.substr(0,scriptDirectory.replace(/[?#].*/,"").lastIndexOf("/")+1)}{if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}return fetch(url,{credentials:"same-origin"}).then(response=>{if(response.ok){return response.arrayBuffer()}return Promise.reject(new Error(response.status+" : "+response.url))})}}}else{}var out=Module["print"]||console.log.bind(console);var err=Module["printErr"]||console.error.bind(console);Object.assign(Module,moduleOverrides);moduleOverrides=null;if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];var wasmBinary=Module["wasmBinary"];var WebAssembly={Memory:function(opts){this.buffer=new ArrayBuffer(opts["initial"]*65536)},Module:function(binary){},Instance:function(module,info){this.exports=(
// EMSCRIPTEN_START_ASM
function instantiate(X){function c(d){d.set=function(a,b){this[a]=b};d.get=function(a){return this[a]};return d}var e;var f=new Uint8Array(123);for(var a=25;a>=0;--a){f[48+a]=52+a;f[65+a]=a;f[97+a]=26+a}f[43]=62;f[47]=63;function l(m,n,o){var g,h,a=0,i=n,j=o.length,k=n+(j*3>>2)-(o[j-2]=="=")-(o[j-1]=="=");for(;a<j;a+=4){g=f[o.charCodeAt(a+1)];h=f[o.charCodeAt(a+2)];m[i++]=f[o.charCodeAt(a)]<<2|g>>4;if(i<k)m[i++]=g<<4|h>>2;if(i<k)m[i++]=h<<6|f[o.charCodeAt(a+3)]}}function p(q){l(e,1025,"Y2xvY2tfZ2V0dGltZShDTE9DS19NT05PVE9OSUMpIGZhaWxlZA==")}function r(){throw new Error("abort")}function W(q){var s=new ArrayBuffer(16908288);var t=new Int8Array(s);var u=new Int16Array(s);var v=new Int32Array(s);var w=new Uint8Array(s);var x=new Uint16Array(s);var y=new Uint32Array(s);var z=new Float32Array(s);var A=new Float64Array(s);var B=Math.imul;var C=Math.fround;var D=Math.abs;var E=Math.clz32;var F=Math.min;var G=Math.max;var H=Math.floor;var I=Math.ceil;var J=Math.trunc;var K=Math.sqrt;var L=q.env;var M=L.emscripten_asm_const_int;var N=L.renderFrame;var O=L._emscripten_memcpy_js;var P=L._emscripten_get_now_is_monotonic;var Q=L.emscripten_get_now;var R=L._abort_js;var S=72032;var T=0;
// EMSCRIPTEN_START_FUNCS
function da(a){var b=0,c=0,d=0,e=0,f=0,g=0,h=C(0),i=C(0),j=C(0),k=0,l=C(0),m=0,n=C(0),o=0,p=C(0),q=C(0),r=C(0),s=0,u=0,w=C(0);a:{b:{h=z[898];i=z[904];if(h>C(i+z[906])){break b}n=z[900];if(n<i){break b}l=z[897];if(l>C(i+z[905])){break b}j=z[899];if(j<z[903]){break b}k=v[902];if((k|0)<=0){break a}while(1){d=b<<4;c=d+5228|0;i=z[d+5232>>2];c:{if(!(h>C(i+z[d+5240>>2])|i>n|l>C(i+z[c+8>>2]))){if(!(j<z[c>>2])){break c}}v[c>>2]=1120403456;v[c+4>>2]=-1054867456}b=b+1|0;if((k|0)!=(b|0)){continue}break}g=v[288];s=g&2147483644;k=g&3;i=z[1507];n=z[1508];u=(g|0)<=0;while(1){h=z[904];b=o<<4;l=z[b+5232>>2];j=C(l+z[b+5240>>2]);d:{if(!(h<j)){break d}q=C(h+z[906]);if(!(q>l)){break d}r=z[903];e=b+5228|0;p=z[e>>2];if(!(r<C(p+z[e+8>>2]))|!(p<C(r+z[905]))){break d}b=o<<3;p=z[b+6036>>2];e:{r=h;h=C(a*n);w=j;j=C(a*z[b+6040>>2]);f:{if(C(r-h)>C(w-j)){ga();v[1507]=0;v[1508]=0;t[6436]=0;n=C(0);g:{if(u){break g}c=0;b=0;d=0;if(g>>>0>=4){while(1){f=b<<3;m=f+6036|0;v[m>>2]=0;v[m+4>>2]=0;t[b+6437|0]=0;m=f+6044|0;v[m>>2]=0;v[m+4>>2]=0;t[b+6438|0]=0;m=f+6052|0;v[m>>2]=0;v[m+4>>2]=0;t[b+6439|0]=0;f=f+6060|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6440|0]=0;b=b+4|0;d=d+4|0;if((s|0)!=(d|0)){continue}break}}if(!k){break g}while(1){d=(b<<3)+6036|0;v[d>>2]=0;v[d+4>>2]=0;t[b+6437|0]=0;b=b+1|0;c=c+1|0;if((k|0)!=(c|0)){continue}break}}h=C(0);break f}if(C(q-h)<C(l-j)){break e}h=i}l=z[903];j=C(a*i);q=z[e>>2];p=C(a*p);h:{if(C(l-j)>C(C(q+z[e+8>>2])-p)){ga();v[1507]=0;v[1508]=0;t[6436]=0;n=C(0);if(u){break h}c=0;b=0;d=0;if(g>>>0>=4){while(1){e=b<<3;f=e+6036|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6437|0]=0;f=e+6044|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6438|0]=0;f=e+6052|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6439|0]=0;e=e+6060|0;v[e>>2]=0;v[e+4>>2]=0;t[b+6440|0]=0;b=b+4|0;d=d+4|0;if((s|0)!=(d|0)){continue}break}}if(!k){break h}while(1){d=(b<<3)+6036|0;v[d>>2]=0;v[d+4>>2]=0;t[b+6437|0]=0;b=b+1|0;c=c+1|0;if((k|0)!=(c|0)){continue}break}break h}i=h;if(!(C(C(l+z[905])-j)<C(q-p))){break d}ga();v[1507]=0;v[1508]=0;t[6436]=0;n=C(0);if(u){break h}c=0;b=0;d=0;if(g>>>0>=4){while(1){e=b<<3;f=e+6036|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6437|0]=0;f=e+6044|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6438|0]=0;f=e+6052|0;v[f>>2]=0;v[f+4>>2]=0;t[b+6439|0]=0;e=e+6060|0;v[e>>2]=0;v[e+4>>2]=0;t[b+6440|0]=0;b=b+4|0;d=d+4|0;if((s|0)!=(d|0)){continue}break}}if(!k){break h}while(1){d=(b<<3)+6036|0;v[d>>2]=0;v[d+4>>2]=0;t[b+6437|0]=0;b=b+1|0;c=c+1|0;if((k|0)!=(c|0)){continue}break}}i=C(0);break d}v[e>>2]=1120403456;v[e+4>>2]=-1054867456}o=o+1|0;if((o|0)<v[902]){continue}break}break a}ga();v[1507]=0;v[1508]=0;t[6436]=0;c=v[288];if((c|0)<=0){break a}d=c&3;if(c>>>0>=4){k=c&2147483644;while(1){c=b<<3;g=c+6036|0;v[g>>2]=0;v[g+4>>2]=0;t[b+6437|0]=0;g=c+6044|0;v[g>>2]=0;v[g+4>>2]=0;t[b+6438|0]=0;g=c+6052|0;v[g>>2]=0;v[g+4>>2]=0;t[b+6439|0]=0;c=c+6060|0;v[c>>2]=0;v[c+4>>2]=0;t[b+6440|0]=0;b=b+4|0;e=e+4|0;if((k|0)!=(e|0)){continue}break}}if(!d){break a}while(1){c=(b<<3)+6036|0;v[c>>2]=0;v[c+4>>2]=0;t[b+6437|0]=0;b=b+1|0;o=o+1|0;if((d|0)!=(o|0)){continue}break}}}function fa(){var a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0,i=0,j=0,k=0,l=0,m=0,n=0,o=0,p=0,q=0,r=0,s=C(0),t=0,u=0,x=0;p=S-16|0;S=p;a=ia();b=T;while(1){m=ia();n=m-a|0;r=+(n>>>0);n=T;s=C(r+ +(n-((a>>>0>m>>>0)+b|0)|0)*4294967296);z[1507]=C(z[895]*s)+z[1507];if(w[6436]==1){z[1508]=z[1508]-z[896]}ca(s);da(s);N();a=ia();b=m-a|0;a=n-(T+(a>>>0>m>>>0)|0)|0;d=a+1|0;c=a;a=b+33e6|0;b=a>>>0<33e6?d:c;v[p+8>>2]=a;v[p+12>>2]=b;l=S-16|0;S=l;a=S-16|0;S=a;v[a>>2]=0;v[a+4>>2]=0;b=ha(a+8|0,a);c=v[b>>2];b=v[b+4>>2];S=a+16|0;v[l+8>>2]=c;v[l+12>>2]=b;a=S-16|0;S=a;j=p+8|0;b=j;c=v[b+4>>2];v[a+8>>2]=v[b>>2];v[a+12>>2]=c;c=v[a+8>>2];b=v[a+12>>2];f=v[l+12>>2];v[a>>2]=v[l+8>>2];v[a+4>>2]=f;S=a+16|0;f=v[a>>2];a=v[a+4>>2];if((f>>>0<c>>>0&(a|0)<=(b|0)|(a|0)<(b|0))-(c>>>0<f>>>0&(a|0)>=(b|0)|(a|0)>(b|0))<<24>>24>0){f=S-16|0;S=f;g=S-16|0;S=g;t=S-16|0;S=t;c=S-16|0;S=c;d=v[j+4>>2];b=d>>31;e=b^v[j>>2];a=e-b|0;o=0;d=(b^d)-((b>>>0>e>>>0)+b|0)|0;a:{if(!d){T=0;a=(a>>>0)/1e9|0;break a}h=35-E(d)|0;k=0-h|0;i=h&63;e=i&31;if(i>>>0>=32){i=0;q=d>>>e|0}else{i=d>>>e|0;q=((1<<e)-1&d)<<32-e|a>>>e}k=k&63;e=k&31;if(k>>>0>=32){d=a<<e;a=0}else{d=(1<<e)-1&a>>>32-e|d<<e;a=a<<e}if(h){while(1){i=i<<1|q>>>31;e=q<<1|d>>>31;k=0-(i+(e>>>0>999999999)|0)>>31;u=k&1e9;q=e-u|0;i=i-(e>>>0<u>>>0)|0;d=d<<1|a>>>31;a=o|a<<1;o=k&1;h=h-1|0;if(h){continue}break}}T=d<<1|a>>>31;a=o|a<<1}a=b^a;d=a-b|0;T=(T^b)-((a>>>0<b>>>0)+b|0)|0;v[c>>2]=d;v[c+4>>2]=T;a=ha(c+8|0,c);b=v[a>>2];a=v[a+4>>2];S=c+16|0;S=t+16|0;v[g+8>>2]=b;v[g+12>>2]=a;a=-1;b=2147483647;d=g+8|0;c=d;if(v[c>>2]!=-1|v[c+4>>2]!=2147483647){a=v[c>>2];b=v[c+4>>2];c=S-32|0;S=c;e=v[j+4>>2];v[c+8>>2]=v[j>>2];v[c+12>>2]=e;e=v[c+12>>2];j=v[c+8>>2];d=ja(c,d);h=v[d>>2];o=j-h|0;d=e-(v[d+4>>2]+(j>>>0<h>>>0)|0)|0;v[c+16>>2]=o;v[c+20>>2]=d;d=ha(c+24|0,c+16|0);e=v[d>>2];d=v[d+4>>2];S=c+32|0;v[g>>2]=e;v[g+4>>2]=d;c=v[g>>2]}else{c=999999999}v[f+8>>2]=c;v[f>>2]=a;v[f+4>>2]=b;S=g+16|0;while(1){a=S-16|0;S=a;b=28;b:{if(!f){break b}c=v[f+8>>2];if(c>>>0>999999999){break b}g=v[f+4>>2];if((g|0)<0){break b}r=(+y[f>>2]+ +(g|0)*4294967296)*1e3+ +(c|0)/1e6;x=+Q();while(1){if(+Q()-x<r){continue}break}b=0}S=a+16|0;a=0-b|0;if(a>>>0>=4294963201){v[1622]=0-a;a=-1}if((a|0)==-1&v[1622]==27){continue}break}S=f+16|0}S=l+16|0;a=m;b=n;continue}}function ba(a){var b=C(0),c=C(0),d=0,e=C(0),f=0,g=C(0),h=C(0),i=C(0),j=C(0),k=C(0),l=0,m=0,n=C(0),o=C(0),p=C(0),q=0,r=0,s=C(0),u=C(0),w=0,x=0,y=0;t[6436]=0;r=v[901];if((r|0)>0){h=z[905];i=z[906];b=z[904];c=z[903];n=z[1507];s=z[1508];while(1){d=f<<4;e=z[d+3632>>2];j=C(e+z[d+3640>>2]);a:{if(!(j>b)){break a}o=C(b+i);if(!(o>e)){break a}d=d+3628|0;g=z[d>>2];k=C(g+z[d+8>>2]);if(!(k>c)){break a}p=C(c+h);if(!(p>g)){break a}u=C(a*s);b:{if(j<C(b-u)){b=C(j+C(9999999747378752e-21))}else{if(!(e>C(o-u))){break b}t[6436]=1;b=C(C(e-i)+C(-9999999747378752e-21))}z[904]=b;v[1508]=0;s=C(0)}e=C(a*n);c:{if(k<C(c-e)){v[1507]=0;c=C(k+C(9999999747378752e-21));break c}if(!(g>C(p-e))){break a}v[1507]=0;c=C(C(g-h)+C(-9999999747378752e-21))}z[903]=c;n=C(0)}f=f+1|0;if((r|0)!=(f|0)){continue}break}}w=v[902];if((w|0)>0){s=z[905];y=(r|0)<=0;while(1){x=q+6437|0;t[x|0]=0;if(!y){l=(q<<3)+6036|0;d=(q<<4)+5228|0;b=z[d+4>>2];f=0;while(1){m=f<<4;c=z[m+3632>>2];g=C(c+z[m+3640>>2]);d:{if(!(g>b)){break d}j=z[d+12>>2];k=C(b+j);if(!(k>c)){break d}h=z[d>>2];m=m+3628|0;e=z[m>>2];i=C(e+z[m+8>>2]);if(!(h<i)){break d}o=C(h+z[d+8>>2]);if(!(o>e)){break d}n=C(a*z[l>>2]);p=C(a*z[l+4>>2]);e:{if(g<C(b-p)){v[l+4>>2]=0;b=C(g+C(9999999747378752e-21));z[d+4>>2]=b;break e}if(!(c>C(k-p))){break e}v[l+4>>2]=0;b=C(C(c-j)+C(-9999999747378752e-21));z[d+4>>2]=b;t[x|0]=1}f:{if(i<C(h-n)){v[l>>2]=0;c=C(i+C(9999999747378752e-21));break f}if(!(e>C(o-n))){break d}v[l>>2]=0;c=C(C(e-s)+C(-9999999747378752e-21))}z[d>>2]=c}f=f+1|0;if((r|0)!=(f|0)){continue}break}}q=q+1|0;if((w|0)!=(q|0)){continue}break}}}function ca(a){var b=0,c=0,d=0,e=C(0),f=0,g=C(0),h=0,i=0,j=0,k=0,l=C(0),m=C(0);g=C(a*z[893]);l=C(z[1508]+g);i=v[902];h=(i|0)<=0;a:{if(h){break a}f=i&3;if(i>>>0>=4){k=i&2147483644;while(1){c=b<<3;d=c+6040|0;z[d>>2]=g+z[d>>2];d=c+6048|0;z[d>>2]=g+z[d>>2];d=c+6056|0;z[d>>2]=g+z[d>>2];c=c+6064|0;z[c>>2]=g+z[c>>2];b=b+4|0;j=j+4|0;if((k|0)!=(j|0)){continue}break}}if(!f){break a}c=0;while(1){j=(b<<3)+6040|0;z[j>>2]=g+z[j>>2];b=b+1|0;c=c+1|0;if((f|0)!=(c|0)){continue}break}}g=z[894];l=C(l-C(l*g));z[1508]=l;m=z[1507];m=C(m-C(m*g));z[1507]=m;b:{c:{if(!h){f=i&1;h=i-1|0;if(!h){c=0;break c}k=i&2147483646;c=0;j=0;while(1){b=c<<3;d=b+6036|0;e=z[d>>2];z[d>>2]=e-C(e*g);d=b+6040|0;e=z[d>>2];z[d>>2]=e-C(e*g);d=b+6044|0;e=z[d>>2];z[d>>2]=e-C(e*g);b=b+6048|0;e=z[b>>2];z[b>>2]=e-C(e*g);c=c+2|0;j=j+2|0;if((k|0)!=(j|0)){continue}break}break c}z[903]=C(m*a)+z[903];z[904]=C(l*a)+z[904];break b}if(f){b=c<<3;c=b+6036|0;e=z[c>>2];z[c>>2]=e-C(e*g);b=b+6040|0;e=z[b>>2];z[b>>2]=e-C(e*g)}b=0;z[903]=C(m*a)+z[903];z[904]=C(l*a)+z[904];if(h){j=i&2147483646;c=0;while(1){f=b<<4;h=f+5228|0;k=b<<3;z[h>>2]=C(z[k+6036>>2]*a)+z[h>>2];f=f+5232|0;z[f>>2]=C(z[k+6040>>2]*a)+z[f>>2];f=b|1;h=f<<4;k=h+5228|0;f=f<<3;z[k>>2]=C(z[f+6036>>2]*a)+z[k>>2];h=h+5232|0;z[h>>2]=C(z[f+6040>>2]*a)+z[h>>2];b=b+2|0;c=c+2|0;if((j|0)!=(c|0)){continue}break}}if(!(i&1)){break b}c=b<<4;i=c+5228|0;b=b<<3;z[i>>2]=C(z[b+6036>>2]*a)+z[i>>2];c=c+5232|0;z[c>>2]=C(z[b+6040>>2]*a)+z[c>>2];ba(a);return}ba(a)}function _(a,b,c,d){a=C(a);b=C(b);c=C(c);d=C(d);var e=0,f=0,g=0,h=C(0),i=0,j=0,k=C(0),l=C(0),m=0,n=0;z[896]=d;z[895]=c;z[894]=b;z[893]=a;d=C(Infinity);c=C(-Infinity);i=v[287];a:{if((i|0)<=0){b=C(-Infinity);a=C(Infinity);break a}b=C(-Infinity);a=C(Infinity);while(1){f=e<<4;k=z[f+1172>>2];h=C(k+z[f+1180>>2]);b=b<h?C(h+C(20)):b;h=z[f+1176>>2];l=C(h+z[f+1184>>2]);c=c<l?C(l+C(20)):c;a=a>k?C(k+C(-20)):a;d=d>h?C(h+C(-20)):d;e=e+1|0;if((i|0)!=(e|0)){continue}break}b=C(b+C(20));c=C(c+C(20));d=C(d+C(-20));a=C(a+C(-20))}f=0;z[900]=c;z[898]=d;z[899]=b;z[897]=a;m=ga();v[1507]=0;v[1508]=0;t[6436]=0;e=v[288];b:{if((e|0)<=0){break b}i=e&3;if(e>>>0>=4){n=e&2147483644;while(1){e=f<<3;g=e+6036|0;v[g>>2]=0;v[g+4>>2]=0;t[f+6437|0]=0;g=e+6044|0;v[g>>2]=0;v[g+4>>2]=0;t[f+6438|0]=0;g=e+6052|0;v[g>>2]=0;v[g+4>>2]=0;t[f+6439|0]=0;e=e+6060|0;v[e>>2]=0;v[e+4>>2]=0;t[f+6440|0]=0;f=f+4|0;j=j+4|0;if((n|0)!=(j|0)){continue}break}}if(!i){break b}e=0;while(1){j=(f<<3)+6036|0;v[j>>2]=0;v[j+4>>2]=0;t[f+6437|0]=0;f=f+1|0;e=e+1|0;if((i|0)!=(e|0)){continue}break}}return m|0}function ia(){var a=0,b=0,c=0,d=0,e=0,f=0,g=0;c=S-48|0;S=c;b=c+24|0;if(!w[6492]){a=P()|0;t[6492]=1;t[6493]=a}a:{b:{if(w[6493]==1){g=+Q();break b}v[1622]=28;a=-1;break a}d=g/1e3;c:{if(D(d)<0x8000000000000000){e=~~d>>>0;if(D(d)>=1){a=~~(d>0?F(H(d*2.3283064365386963e-10),4294967295):I((d-+(~~d>>>0>>>0))*2.3283064365386963e-10))>>>0}else{a=0}break c}a=-2147483648}v[b>>2]=e;v[b+4>>2]=a;d=(g-(+(na(e,a,1e3)>>>0)+ +(T|0)*4294967296))*1e3*1e3;d:{if(D(d)<2147483648){a=~~d;break d}a=-2147483648}v[b+8>>2]=a;a=0}if(a){R();r()}a=ha(c+8|0,c+24|0);b=v[c+32>>2];v[c>>2]=b;v[c+4>>2]=b>>31;b=S-32|0;S=b;a=ja(b+8|0,a);f=v[a>>2];e=v[a+4>>2];a=v[c+4>>2];v[b>>2]=v[c>>2];v[b+4>>2]=a;e=v[b+4>>2]+e|0;a=v[b>>2];f=f+a|0;v[b+16>>2]=f;v[b+20>>2]=a>>>0>f>>>0?e+1|0:e;a=ha(b+24|0,b+16|0);e=v[a>>2];a=v[a+4>>2];S=b+32|0;v[c+16>>2]=e;v[c+20>>2]=a;a=ha(c+40|0,c+16|0);b=v[a>>2];S=c+48|0;T=v[a+4>>2];return b}function ja(a,b){var c=0,d=0,e=0,f=0;c=S-16|0;S=c;e=S-16|0;S=e;d=S-16|0;S=d;v[d>>2]=na(v[b>>2],v[b+4>>2],1e9);v[d+4>>2]=T;b=ha(d+8|0,d);f=v[b>>2];b=v[b+4>>2];S=d+16|0;S=e+16|0;v[c+8>>2]=f;v[c+12>>2]=b;b=v[c+12>>2];v[a>>2]=v[c+8>>2];v[a+4>>2]=b;S=c+16|0;return a}function na(a,b,c){var d=0,e=0,f=0,g=0,h=0;e=c>>>16|0;d=a>>>16|0;h=B(e,d);f=c&65535;a=a&65535;g=B(f,a);d=(g>>>16|0)+B(d,f)|0;a=(d&65535)+B(a,e)|0;T=h+B(b,c)+(d>>>16)+(a>>>16)|0;return g&65535|a<<16}
function ea(a,b,c){a=C(a);b=C(b);c=C(c);z[1507]=C(C(b*z[895])*a)+z[1507];if(!(!(c>C(0))|w[6436]!=1)){z[1508]=z[1508]-z[896]}ca(a);da(a)}function ha(a,b){var c=0;c=v[b+4>>2];v[a>>2]=v[b>>2];v[a+4>>2]=c;return a}function la(a){a=a|0;a=S-a&-16;S=a;return a|0}function ga(){O(3604,1148,2424);return 3604}function aa(){M(1089,1024,0)|0}function $(){M(1063,1024,0)|0}function ma(){return S|0}function ka(a){a=a|0;S=a}function Z(){return 1148}function Y(){}
// EMSCRIPTEN_END_FUNCS
e=w;p(q);var U=c([]);function V(){return s.byteLength/65536|0}return{memory:Object.create(Object.prototype,{grow:{},buffer:{get:function(){return s}}}),__wasm_call_ctors:Y,getInitialPointer:Z,initializeEnvironment:_,test_func1:$,test_func2:aa,doUpdate:ea,startUpdates:fa,__indirect_function_table:U,_emscripten_stack_restore:ka,_emscripten_stack_alloc:la,emscripten_stack_get_current:ma}}return W(X)}
// EMSCRIPTEN_END_ASM


)(info)},instantiate:function(binary,info){return{then:function(ok){var module=new WebAssembly.Module(binary);ok({instance:new WebAssembly.Instance(module,info)})}}},RuntimeError:Error,isWasm2js:true};if(WebAssembly.isWasm2js){wasmBinary=[]}var wasmMemory;var ABORT=false;var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;function updateMemoryViews(){var b=wasmMemory.buffer;Module["HEAP8"]=HEAP8=new Int8Array(b);Module["HEAP16"]=HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);Module["HEAPU16"]=HEAPU16=new Uint16Array(b);Module["HEAP32"]=HEAP32=new Int32Array(b);Module["HEAPU32"]=HEAPU32=new Uint32Array(b);Module["HEAPF32"]=HEAPF32=new Float32Array(b);Module["HEAPF64"]=HEAPF64=new Float64Array(b)}var __ATPRERUN__=[];var __ATINIT__=[];var __ATPOSTRUN__=[];var runtimeInitialized=false;function preRun(){var preRuns=Module["preRun"];if(preRuns){if(typeof preRuns=="function")preRuns=[preRuns];preRuns.forEach(addOnPreRun)}callRuntimeCallbacks(__ATPRERUN__)}function initRuntime(){runtimeInitialized=true;callRuntimeCallbacks(__ATINIT__)}function postRun(){var postRuns=Module["postRun"];if(postRuns){if(typeof postRuns=="function")postRuns=[postRuns];postRuns.forEach(addOnPostRun)}callRuntimeCallbacks(__ATPOSTRUN__)}function addOnPreRun(cb){__ATPRERUN__.unshift(cb)}function addOnInit(cb){__ATINIT__.unshift(cb)}function addOnPostRun(cb){__ATPOSTRUN__.unshift(cb)}var runDependencies=0;var runDependencyWatcher=null;var dependenciesFulfilled=null;function addRunDependency(id){runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)}function removeRunDependency(id){runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(runDependencyWatcher!==null){clearInterval(runDependencyWatcher);runDependencyWatcher=null}if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";var e=new WebAssembly.RuntimeError(what);readyPromiseReject(e);throw e}var dataURIPrefix="data:application/octet-stream;base64,";var isDataURI=filename=>filename.startsWith(dataURIPrefix);var isFileURI=filename=>filename.startsWith("file://");function findWasmBinary(){if(Module["locateFile"]){var f="wasm.wasm";if(!isDataURI(f)){return locateFile(f)}return f}return new URL("wasm.wasm",import.meta.url).href}var wasmBinaryFile;function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}function getBinaryPromise(binaryFile){if(!wasmBinary){return readAsync(binaryFile).then(response=>new Uint8Array(response),()=>getBinarySync(binaryFile))}return Promise.resolve().then(()=>getBinarySync(binaryFile))}function instantiateArrayBuffer(binaryFile,imports,receiver){return getBinaryPromise(binaryFile).then(binary=>WebAssembly.instantiate(binary,imports)).then(receiver,reason=>{err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)})}function instantiateAsync(binary,binaryFile,imports,callback){if(!binary&&typeof WebAssembly.instantiateStreaming=="function"&&!isDataURI(binaryFile)&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE&&typeof fetch=="function"){return fetch(binaryFile,{credentials:"same-origin"}).then(response=>{var result=WebAssembly.instantiateStreaming(response,imports);return result.then(callback,function(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation");return instantiateArrayBuffer(binaryFile,imports,callback)})})}return instantiateArrayBuffer(binaryFile,imports,callback)}function getWasmImports(){return{env:wasmImports,wasi_snapshot_preview1:wasmImports}}function createWasm(){var info=getWasmImports();function receiveInstance(instance,module){wasmExports=instance.exports;wasmMemory=wasmExports["memory"];updateMemoryViews();addOnInit(wasmExports["__wasm_call_ctors"]);removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){receiveInstance(result["instance"])}if(Module["instantiateWasm"]){try{return Module["instantiateWasm"](info,receiveInstance)}catch(e){err(`Module.instantiateWasm callback failed with error: ${e}`);readyPromiseReject(e)}}wasmBinaryFile??=findWasmBinary();instantiateAsync(wasmBinary,wasmBinaryFile,info,receiveInstantiationResult).catch(readyPromiseReject);return{}}var ASM_CONSTS={1063:()=>{console.log("test1:wasm")},1089:()=>{console.log("test2:wasm")}};function renderFrame(){renderEnvironment()}var callRuntimeCallbacks=callbacks=>{callbacks.forEach(f=>f(Module))};var noExitRuntime=Module["noExitRuntime"]||true;var __abort_js=()=>{abort("")};var nowIsMonotonic=1;var __emscripten_get_now_is_monotonic=()=>nowIsMonotonic;var __emscripten_memcpy_js=(dest,src,num)=>HEAPU8.copyWithin(dest,src,src+num);var readEmAsmArgsArray=[];var readEmAsmArgs=(sigPtr,buf)=>{readEmAsmArgsArray.length=0;var ch;while(ch=HEAPU8[sigPtr++]){var wide=ch!=105;wide&=ch!=112;buf+=wide&&buf%8?4:0;readEmAsmArgsArray.push(ch==112?HEAPU32[buf>>2]:ch==105?HEAP32[buf>>2]:HEAPF64[buf>>3]);buf+=wide?8:4}return readEmAsmArgsArray};var runEmAsmFunction=(code,sigPtr,argbuf)=>{var args=readEmAsmArgs(sigPtr,argbuf);return ASM_CONSTS[code](...args)};var _emscripten_asm_const_int=(code,sigPtr,argbuf)=>runEmAsmFunction(code,sigPtr,argbuf);var _emscripten_date_now=()=>Date.now();var _emscripten_get_now=()=>performance.now();var wasmImports={_abort_js:__abort_js,_emscripten_get_now_is_monotonic:__emscripten_get_now_is_monotonic,_emscripten_memcpy_js:__emscripten_memcpy_js,emscripten_asm_const_int:_emscripten_asm_const_int,emscripten_date_now:_emscripten_date_now,emscripten_get_now:_emscripten_get_now,renderFrame};var wasmExports=createWasm();var ___wasm_call_ctors=()=>(___wasm_call_ctors=wasmExports["__wasm_call_ctors"])();var _getInitialPointer=Module["_getInitialPointer"]=()=>(_getInitialPointer=Module["_getInitialPointer"]=wasmExports["getInitialPointer"])();var _initializeEnvironment=Module["_initializeEnvironment"]=(a0,a1,a2,a3)=>(_initializeEnvironment=Module["_initializeEnvironment"]=wasmExports["initializeEnvironment"])(a0,a1,a2,a3);var _test_func1=Module["_test_func1"]=()=>(_test_func1=Module["_test_func1"]=wasmExports["test_func1"])();var _test_func2=Module["_test_func2"]=()=>(_test_func2=Module["_test_func2"]=wasmExports["test_func2"])();var _doUpdate=Module["_doUpdate"]=(a0,a1,a2)=>(_doUpdate=Module["_doUpdate"]=wasmExports["doUpdate"])(a0,a1,a2);var _startUpdates=Module["_startUpdates"]=()=>(_startUpdates=Module["_startUpdates"]=wasmExports["startUpdates"])();var __emscripten_stack_restore=a0=>(__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"])(a0);var __emscripten_stack_alloc=a0=>(__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"])(a0);var _emscripten_stack_get_current=()=>(_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"])();var calledRun;var calledPrerun;dependenciesFulfilled=function runCaller(){if(!calledRun)run();if(!calledRun)dependenciesFulfilled=runCaller};function run(){if(runDependencies>0){return}if(!calledPrerun){calledPrerun=1;preRun();if(runDependencies>0){return}}function doRun(){if(calledRun)return;calledRun=1;Module["calledRun"]=1;if(ABORT)return;initRuntime();readyPromiseResolve(Module);Module["onRuntimeInitialized"]?.();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].pop()()}}run();moduleRtn=readyPromise;


  return moduleRtn;
}
);
})();
export default Module;