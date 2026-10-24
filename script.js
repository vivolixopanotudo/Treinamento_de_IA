'use strict';
const $ = id => document.getElementById(id);
let model = null;
let names = [];
let size = 96;
let stream = null;
let imageReady = false;
const status = text => { $('status').textContent = text; };
function stopCamera(){
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  $('video').hidden = true;
  $('stopCamera').disabled = true;
  $('takePhoto').disabled = true;
}
function setImage(url){
  stopCamera();
  const img=$('previewImage');
  imageReady=false; $('classify').disabled=true;
  img.onload=()=>{imageReady=true;$('classify').disabled=!model;status('Imagem pronta para classificar.');};
  img.onerror=()=>status('Não foi possível abrir esta imagem.');
  img.src=url; img.hidden=false; $('placeholder').hidden=true;
  $('result').textContent='Clique em “Classificar imagem”.'; $('scores').replaceChildren();
}
function readFile(file){
  if (!file) return;
  if (!file.type.startsWith('image/')) {status('Escolha um arquivo de imagem.');return;}
  const reader=new FileReader(); reader.onload=()=>setImage(reader.result); reader.onerror=()=>status('Falha ao ler imagem.');reader.readAsDataURL(file);
}
$('gallery').addEventListener('change', e=>{readFile(e.target.files[0]);e.target.value='';});
$('photo').addEventListener('change', e=>{readFile(e.target.files[0]);e.target.value='';});
$('startCamera').addEventListener('click',async()=>{
  if (!navigator.mediaDevices?.getUserMedia){status('Câmera ao vivo indisponível. Use “Tirar foto” ou “Escolher imagem”.');return;}
  try{
    stopCamera();
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
    $('video').srcObject=stream; $('video').hidden=false; $('previewImage').hidden=true; $('placeholder').hidden=true;
    $('classify').disabled=true; imageReady=false;
    $('takePhoto').disabled=false; $('stopCamera').disabled=false;
    status('Câmera ligada. Clique em “Capturar da câmera”.');
  }catch(err){status('Não foi possível abrir a câmera: '+err.message+' — use a galeria se preferir.');}
});
$('takePhoto').addEventListener('click',()=>{
  const video=$('video');
  if (!stream || !video.videoWidth){status('Aguarde a câmera carregar.');return;}
  const canvas=$('captureCanvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  canvas.getContext('2d').drawImage(video,0,0);
  setImage(canvas.toDataURL('image/jpeg',0.92));
});
$('stopCamera').addEventListener('click',()=>{stopCamera();$('placeholder').hidden=!$('previewImage').hidden;status('Câmera desligada.');});
window.addEventListener('pagehide',stopCamera);
$('classify').addEventListener('click',async()=>{
  if(!model || !imageReady)return;
  $('classify').disabled=true; status('Classificando…');
  try{
    // Mesmo pré-processamento do Colab v2: RGB, 96 × 96, valores / 255.
    const scores=await tf.tidy(()=>{
      const pixels=tf.browser.fromPixels($('previewImage'),3);
      const batch=tf.image.resizeBilinear(pixels,[size,size]).toFloat().div(255).expandDims(0);
      const out=model.predict(batch);
      return Array.from(out.dataSync());
    });
    if(scores.length!==names.length)throw new Error(`Modelo possui ${scores.length} saídas, mas classes.json possui ${names.length} categorias.`);
    const ranked=scores.map((p,i)=>({p,name:names[i]})).sort((a,b)=>b.p-a.p);
    $('result').textContent=`Classe mais provável: ${ranked[0].name} (${(ranked[0].p*100).toFixed(1)}%)`;
    $('scores').replaceChildren();
    for(const item of ranked){
      const row=document.createElement('div');row.className='score';
      const head=document.createElement('div');head.className='scoreHead';
      const title=document.createElement('span');title.textContent=item.name;
      const number=document.createElement('span');number.textContent=(item.p*100).toFixed(1)+'%';
      head.append(title,number);const track=document.createElement('div');track.className='track';
      const fill=document.createElement('div');fill.className='fill';fill.style.width=Math.min(100,Math.max(0,item.p*100))+'%';
      track.append(fill);row.append(head,track);$('scores').append(row);
    }
    status('Classificação concluída.');
  }catch(err){status('Erro de classificação: '+err.message);console.error(err);}
  finally{$('classify').disabled=!imageReady||!model;}
});
(async()=>{
  try{
    if(typeof tf==='undefined')throw new Error('Biblioteca TensorFlow.js não carregou. Verifique sua conexão com a internet.');
    const response=await fetch('./model/classes.json');if(!response.ok)throw new Error(`classes.json: HTTP ${response.status}`);
    const classes=await response.json();
    if(!Array.isArray(classes.classes)||!classes.classes.length)throw new Error('classes.json não contém lista "classes".');
    names=classes.classes;size=Number(classes.image_size)||96;
    model=await tf.loadLayersModel('./model/model.json');
    status(`Modelo carregado: ${names.length} categorias. Escolha uma foto.`);
    $('classify').disabled=!imageReady;
  }catch(err){status('Modelo ainda não disponível: '+err.message);console.error(err);}
})();
