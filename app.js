const $ = (id) => document.getElementById(id);
const fileInput=$('fileInput'), preview=$('preview'), emptyState=$('emptyState'), analyzeBtn=$('analyzeBtn'), resetBtn=$('resetBtn'), dropzone=$('dropzone'), fileMeta=$('fileMeta');
const placeholder=$('placeholder'), loading=$('loading'), result=$('result'), loadingText=$('loadingText'), modelPill=$('modelPill');
let selectedFile=null, previewUrl=null, MODEL=null, modelReady=false;

const LABEL_META = {
  "opacity": {display:"Opacity", description:"Model menemukan pola yang menyerupai label opacity/kekeruhan pada dataset pelatihan.", next:"Disarankan pemeriksaan mata langsung untuk memastikan lokasi dan penyebab kekeruhan."},
  "diabetic retinopathy": {display:"Retinopati Diabetik", description:"Model menemukan pola citra yang menyerupai retinopati diabetik pada dataset.", next:"Bila pasien memiliki diabetes atau keluhan penglihatan, pertimbangkan evaluasi dokter mata dan pemeriksaan retina."},
  "glaucoma": {display:"Glaukoma", description:"Model menemukan pola yang pada dataset dikaitkan dengan glaukoma.", next:"Glaukoma tidak dapat dipastikan dari model ini saja. Pemeriksaan tekanan intraokular, saraf optik, dan lapang pandang tetap diperlukan."},
  "macular edema": {display:"Edema Makula", description:"Model menemukan pola yang menyerupai edema pada area makula.", next:"Pertimbangkan evaluasi dokter mata; pemeriksaan tambahan seperti OCT dapat diperlukan sesuai penilaian klinis."},
  "macular degeneration": {display:"Degenerasi Makula", description:"Model menemukan pola yang menyerupai degenerasi makula pada dataset pelatihan.", next:"Disarankan pemeriksaan retina/makula oleh dokter mata untuk konfirmasi dan penilaian derajat kelainan."},
  "retinal vascular occlusion": {display:"Oklusi Vaskular Retina", description:"Model menemukan pola yang menyerupai oklusi pembuluh darah retina.", next:"Keluhan penurunan penglihatan mendadak perlu dinilai segera oleh tenaga medis atau dokter mata."},
  "normal": {display:"Normal", description:"Model lebih mendukung label normal berdasarkan pola yang dipelajari dari dataset.", next:"Hasil normal dari AI tidak menyingkirkan seluruh penyakit mata. Tetap periksakan mata bila ada keluhan atau faktor risiko."}
};

function tensor(meta, buffer, name){
  const t=meta.tensors[name];
  return new Float32Array(buffer, t.offset, t.length);
}

async function loadModel(){
  try{
    modelPill.innerHTML='<i></i>Memuat CNN…';
    const [metaRes, binRes] = await Promise.all([fetch('./model.json'), fetch('./model.bin')]);
    if(!metaRes.ok || !binRes.ok) throw new Error('File model tidak dapat dimuat.');
    const meta=await metaRes.json();
    const buffer=await binRes.arrayBuffer();
    const T={};
    Object.keys(meta.tensors).forEach(name=>T[name]=tensor(meta,buffer,name));
    MODEL={meta,T}; modelReady=true;
    modelPill.className='model-pill active'; modelPill.innerHTML='<i></i>CNN aktif · browser';
    analyzeBtn.disabled=!selectedFile;
  }catch(err){
    console.error(err); modelReady=false;
    modelPill.className='model-pill inactive'; modelPill.innerHTML='<i></i>Model gagal dimuat';
  }
}
loadModel();

function humanSize(bytes){ return bytes < 1024*1024 ? `${(bytes/1024).toFixed(0)} KB` : `${(bytes/1024/1024).toFixed(2)} MB`; }
function setFile(file){
  if(!file || !file.type.startsWith('image/')) return;
  if(file.size > 10*1024*1024){ alert('Ukuran file maksimal 10 MB.'); return; }
  selectedFile=file;
  if(previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(file); preview.src=previewUrl; preview.hidden=false; emptyState.hidden=true;
  fileMeta.textContent=`${file.name} · ${humanSize(file.size)}`; analyzeBtn.disabled=!modelReady; placeholder.hidden=false; loading.hidden=true; result.hidden=true;
}
fileInput.addEventListener('change',e=>setFile(e.target.files[0]));
['dragenter','dragover'].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.classList.add('drag')}));
['dragleave','drop'].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.classList.remove('drag')}));
dropzone.addEventListener('drop',e=>setFile(e.dataTransfer.files[0]));
resetBtn.addEventListener('click',()=>{
  selectedFile=null;fileInput.value='';if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;preview.src='';preview.hidden=true;emptyState.hidden=false;fileMeta.textContent='Belum ada file dipilih.';analyzeBtn.disabled=true;placeholder.hidden=false;loading.hidden=true;result.hidden=true;
});

function waitFrame(){ return new Promise(r=>requestAnimationFrame(()=>setTimeout(r,0))); }

async function imageToTensor(file, size){
  const bitmap=await createImageBitmap(file);
  const originalW=bitmap.width, originalH=bitmap.height;
  const canvas=document.createElement('canvas'); canvas.width=size; canvas.height=size;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality='high';
  ctx.drawImage(bitmap,0,0,size,size);
  const pix=ctx.getImageData(0,0,size,size).data;
  const out=new Float32Array(3*size*size);
  let graySum=0, n=size*size;
  for(let i=0;i<n;i++){
    const p=i*4, r=pix[p], g=pix[p+1], b=pix[p+2];
    out[i]=r/127.5-1; out[n+i]=g/127.5-1; out[2*n+i]=b/127.5-1;
    graySum += 0.299*r + 0.587*g + 0.114*b;
  }
  bitmap.close?.();
  const mean=graySum/n, warnings=[];
  if(Math.min(originalW,originalH)<128) warnings.push('Resolusi citra rendah; gunakan gambar yang lebih besar bila tersedia.');
  if(mean<35) warnings.push('Citra sangat gelap sehingga prediksi dapat kurang stabil.');
  else if(mean>225) warnings.push('Citra sangat terang sehingga detail retina dapat berkurang.');
  return {data:out, quality:{width:originalW,height:originalH,mean_brightness:+mean.toFixed(1),warnings}};
}

function convBnReluPool(input,inC,h,w,convW,bnW,bnB,mean,variance,outC,eps){
  const oh=h>>1, ow=w>>1, output=new Float32Array(outC*oh*ow);
  const inPlane=h*w, outPlane=oh*ow;
  for(let oc=0;oc<outC;oc++){
    const scale=bnW[oc]/Math.sqrt(variance[oc]+eps);
    const shift=bnB[oc]-mean[oc]*scale;
    for(let py=0;py<oh;py++){
      const y0=py*2;
      for(let px=0;px<ow;px++){
        const x0=px*2; let maxv=0;
        for(let ddy=0;ddy<2;ddy++){
          const y=y0+ddy;
          for(let ddx=0;ddx<2;ddx++){
            const x=x0+ddx; let sum=0;
            for(let ic=0;ic<inC;ic++){
              const ib=ic*inPlane, wb=(oc*inC+ic)*9;
              for(let ky=0;ky<3;ky++){
                const iy=y+ky-1; if(iy<0||iy>=h) continue;
                const row=ib+iy*w;
                for(let kx=0;kx<3;kx++){
                  const ix=x+kx-1; if(ix<0||ix>=w) continue;
                  sum += input[row+ix]*convW[wb+ky*3+kx];
                }
              }
            }
            let v=sum*scale+shift; if(v<0)v=0; if(v>maxv)maxv=v;
          }
        }
        output[oc*outPlane+py*ow+px]=maxv;
      }
    }
  }
  return {data:output,h:oh,w:ow,c:outC};
}

function globalAvg(input,c,h,w){
  const area=h*w, out=new Float32Array(c);
  for(let ch=0;ch<c;ch++){ let s=0, b=ch*area; for(let i=0;i<area;i++)s+=input[b+i]; out[ch]=s/area; }
  return out;
}
function dense(input,weight,bias,outN,inN,relu=false){
  const out=new Float32Array(outN);
  for(let o=0;o<outN;o++){ let s=bias[o], b=o*inN; for(let i=0;i<inN;i++)s+=weight[b+i]*input[i]; out[o]=relu&&s<0?0:s; }
  return out;
}
function sigmoid(x){ return x>=0 ? 1/(1+Math.exp(-x)) : Math.exp(x)/(1+Math.exp(x)); }

async function runInference(file){
  const {meta,T}=MODEL, size=meta.image_size;
  loadingText.textContent='Resize & normalisasi citra…'; await waitFrame();
  const prep=await imageToTensor(file,size);
  let x={data:prep.data,c:3,h:size,w:size};
  const channels=[16,32,64,96];
  for(let b=0;b<4;b++){
    loadingText.textContent=`Mengekstraksi fitur CNN · blok ${b+1}/4…`; await waitFrame();
    x=convBnReluPool(x.data,x.c,x.h,x.w,
      T[`features.${b}.0.weight`],T[`features.${b}.1.weight`],T[`features.${b}.1.bias`],T[`features.${b}.1.running_mean`],T[`features.${b}.1.running_var`],channels[b],meta.eps||1e-5);
  }
  loadingText.textContent='Menghitung 7 sigmoid scores…'; await waitFrame();
  let v=globalAvg(x.data,x.c,x.h,x.w);
  v=dense(v,T['head.2.weight'],T['head.2.bias'],64,96,true);
  const logits=dense(v,T['head.4.weight'],T['head.4.bias'],7,64,false);
  const probs=Array.from(logits,sigmoid);
  return formatResult(probs,prep.quality,meta,file.name);
}

function formatResult(probs,quality,meta,filename){
  const labels=meta.labels, thresholds=meta.thresholds;
  const by=Object.fromEntries(labels.map((n,i)=>[n,probs[i]]));
  const diseases=labels.filter(n=>n!=='normal');
  const positive=diseases.filter(n=>by[n]>=thresholds[n]);
  const normalPositive=by.normal>=thresholds.normal;
  let detected,state,summary;
  if(positive.length){detected=positive;state='perlu-evaluasi';summary=`${positive.length} label penyakit melewati threshold model.`}
  else if(normalPositive){detected=['normal'];state='normal';summary='Label normal melewati threshold dan tidak ada label penyakit yang melewati threshold.'}
  else{detected=[];state='tidak-pasti';summary='Tidak ada label yang melewati threshold; hasil perlu dianggap tidak pasti.'}
  const ranked=labels.slice().sort((a,b)=>by[b]-by[a]);
  return {
    filename,state,summary,quality,
    top_label:LABEL_META[ranked[0]].display,top_probability:by[ranked[0]]*100,
    detected:detected.map(n=>({name:LABEL_META[n].display,probability:by[n]*100,threshold:thresholds[n]*100,description:LABEL_META[n].description,next_step:LABEL_META[n].next})),
    probabilities:labels.map(n=>({name:LABEL_META[n].display,value:by[n]*100,threshold:thresholds[n]*100,positive:detected.includes(n)}))
  };
}

analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile||!modelReady)return;
  placeholder.hidden=true;result.hidden=true;loading.hidden=false;analyzeBtn.disabled=true;
  const t0=performance.now();
  try{
    const data=await runInference(selectedFile);
    render(data);
    console.log(`Inference ${(performance.now()-t0).toFixed(0)} ms`);
  }catch(err){
    console.error(err);result.innerHTML=`<div class="error-box"><b>Analisis gagal</b><p>${escapeHtml(err.message)}</p></div>`;result.hidden=false;
  }finally{loading.hidden=true;analyzeBtn.disabled=false;}
});
function escapeHtml(v){const d=document.createElement('div');d.textContent=v;return d.innerHTML;}
function render(data){
  const stateMap={"normal":["NORMAL","Tidak ada label penyakit melewati threshold"],"perlu-evaluasi":["PERLU EVALUASI","Model menemukan satu atau lebih pola penyakit"],"tidak-pasti":["TIDAK PASTI","Tidak ada label yang cukup kuat"]};
  const [badge,title]=stateMap[data.state]||['HASIL','Hasil model'];$('stateBadge').textContent=badge;$('stateBadge').className=`badge ${data.state}`;$('resultTitle').textContent=title;$('summary').textContent=data.summary;
  $('topProbability').textContent=`${data.top_probability.toFixed(2)}%`;$('topLabel').textContent=data.top_label;
  const qw=$('qualityWarnings');qw.innerHTML='';(data.quality.warnings||[]).forEach(w=>qw.insertAdjacentHTML('beforeend',`<div class="quality-warning">⚠ ${escapeHtml(w)}</div>`));
  const dl=$('detectedList');dl.innerHTML='';
  if(!data.detected.length){dl.innerHTML='<div class="no-detection">Tidak ada label yang melewati threshold model.</div>'}else{
    data.detected.forEach(x=>dl.insertAdjacentHTML('beforeend',`<article class="detected-item"><div><span>${escapeHtml(x.name)}</span><strong>${x.probability.toFixed(2)}%</strong></div><p>${escapeHtml(x.description)}</p><small><b>Tindak lanjut:</b> ${escapeHtml(x.next_step)}</small></article>`));
  }
  const probs=$('probabilities');probs.innerHTML='';
  data.probabilities.slice().sort((a,b)=>b.value-a.value).forEach(x=>{
    const width=Math.max(1,Math.min(100,x.value)), threshold=Math.min(100,x.threshold);
    probs.insertAdjacentHTML('beforeend',`<div class="prob-row ${x.positive?'positive':''}"><div class="prob-name"><span>${escapeHtml(x.name)}</span><b>${x.value.toFixed(2)}%</b></div><div class="track"><div class="fill" style="width:${width}%"></div><i style="left:${threshold}%" title="Threshold ${x.threshold}%"></i></div><small>threshold ${x.threshold.toFixed(1)}%</small></div>`);
  });
  result.hidden=false;
}

// ---------- RETINA-AI UI navigation & prototype referral flow ----------
const views = [...document.querySelectorAll('.app-view')];
function showView(name){
  views.forEach(v=>v.classList.toggle('active', v.id===`view-${name}`));
  window.scrollTo({top:0,behavior:'smooth'});
  if(name==='history') renderHistory();
  if(name==='appointments') renderAppointments();
}
document.addEventListener('click',e=>{
  const target=e.target.closest('[data-view]');
  if(target){
    const name=target.dataset.view;
    if(name==='booking' && target.dataset.doctorId) selectDoctor(target.dataset.doctorId);
    showView(name);
  }
});

const DOCTORS=[
  {id:'d1',initial:'RA',name:'dr. Raka Adinata, Sp.M(K)',specialty:'Onkologi mata',hospital:'Rumah Sakit Mata · Data demo',rating:'5.0',tags:['onkologi']},
  {id:'d2',initial:'AL',name:'dr. Alya Larasati, Sp.M(K)',specialty:'Oftalmologi pediatrik',hospital:'Klinik Mata Anak · Data demo',rating:'4.9',tags:['pediatrik']},
  {id:'d3',initial:'NS',name:'dr. Naufal Satya, Sp.M',specialty:'Retina & vitreus',hospital:'Pusat Mata · Data demo',rating:'4.8',tags:['onkologi']}
];
let doctorFilter='all', selectedDoctorId='d1', selectedDate='', selectedTime='';
const doctorList=document.getElementById('doctorList');
function renderDoctors(){
  if(!doctorList)return;
  const q=(document.getElementById('doctorSearch')?.value||'').toLowerCase();
  const list=DOCTORS.filter(d=>(doctorFilter==='all'||d.tags.includes(doctorFilter)) && `${d.name} ${d.specialty}`.toLowerCase().includes(q));
  doctorList.innerHTML=list.length?list.map(d=>`<article class="doctor-card"><div class="avatar">${d.initial}</div><div><h3>${escapeHtml(d.name)}</h3><p>${escapeHtml(d.specialty)}</p><p>${escapeHtml(d.hospital)}</p><div class="doctor-meta"><span>★ ${d.rating}</span><span>✓ Profil demo</span></div></div><button class="btn primary" type="button" data-view="booking" data-doctor-id="${d.id}">Buat Janji</button></article>`).join(''):'<div class="empty-list">Tidak ada dokter demo yang cocok dengan pencarian.</div>';
}
renderDoctors();
document.getElementById('doctorSearch')?.addEventListener('input',renderDoctors);
document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',()=>{document.querySelectorAll('.chip').forEach(x=>x.classList.remove('active'));c.classList.add('active');doctorFilter=c.dataset.filter;renderDoctors()}));

function selectDoctor(id){
  selectedDoctorId=id; selectedDate=''; selectedTime='';
  const d=DOCTORS.find(x=>x.id===id)||DOCTORS[0];
  const box=document.getElementById('selectedDoctor');
  if(box)box.innerHTML=`<div class="avatar">${d.initial}</div><div><h3>${escapeHtml(d.name)}</h3><p>${escapeHtml(d.specialty)} · ${escapeHtml(d.hospital)}</p></div>`;
  buildDates(); buildTimes();
}
function buildDates(){
  const el=document.getElementById('dateGrid'); if(!el)return;
  const days=[]; const base=new Date();
  for(let i=1;i<=14;i++){const d=new Date(base);d.setDate(base.getDate()+i);days.push(d)}
  el.innerHTML=days.map(d=>{const iso=d.toISOString().slice(0,10);return `<button class="date-option" type="button" data-date="${iso}"><small>${d.toLocaleDateString('id-ID',{weekday:'short'})}</small>${d.getDate()}</button>`}).join('');
  el.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{el.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');selectedDate=b.dataset.date;updateConfirm()}));
}
function buildTimes(){
  const el=document.getElementById('timeGrid'); if(!el)return;
  const times=['08:30','09:00','10:30','11:00','13:00','14:00'];
  el.innerHTML=times.map(t=>`<button class="time-option" type="button" data-time="${t}">${t}</button>`).join('');
  el.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{el.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');selectedTime=b.dataset.time;updateConfirm()}));
}
function updateConfirm(){const b=document.getElementById('confirmAppointment');if(b)b.disabled=!(selectedDate&&selectedTime)}
selectDoctor(selectedDoctorId);
document.getElementById('confirmAppointment')?.addEventListener('click',()=>{
  const doctor=DOCTORS.find(d=>d.id===selectedDoctorId)||DOCTORS[0];
  const data={id:Date.now(),doctor:doctor.name,specialty:doctor.specialty,date:selectedDate,time:selectedTime};
  const arr=JSON.parse(localStorage.getItem('retinaAIAppointments')||'[]');arr.unshift(data);localStorage.setItem('retinaAIAppointments',JSON.stringify(arr));
  alert('Janji temu demo berhasil disimpan di browser.');showView('appointments');
});

function renderAppointments(){
  const el=document.getElementById('appointmentList');if(!el)return;
  const arr=JSON.parse(localStorage.getItem('retinaAIAppointments')||'[]');
  el.innerHTML=arr.length?arr.map(a=>`<article class="history-item"><div><b>${escapeHtml(a.doctor)}</b><small>${escapeHtml(a.specialty)}<br>${new Date(a.date+'T00:00:00').toLocaleDateString('id-ID',{dateStyle:'long'})} · ${escapeHtml(a.time)}</small></div><span class="badge normal">TERJADWAL</span></article>`).join(''):'<div class="empty-list">Belum ada janji temu. Pilih “Cari Spesialis” untuk membuat jadwal demo.</div>';
}
function saveHistory(data){
  try{
    const arr=JSON.parse(localStorage.getItem('retinaAIHistory')||'[]');
    arr.unshift({id:Date.now(),date:new Date().toISOString(),filename:data.filename,state:data.state,top_label:data.top_label,top_probability:data.top_probability});
    localStorage.setItem('retinaAIHistory',JSON.stringify(arr.slice(0,20)));
  }catch(e){console.warn('Riwayat gagal disimpan',e)}
}
function renderHistory(){
  const el=document.getElementById('historyList');if(!el)return;
  const arr=JSON.parse(localStorage.getItem('retinaAIHistory')||'[]');
  el.innerHTML=arr.length?arr.map(h=>`<article class="history-item"><div><b>${escapeHtml(h.filename)}</b><small>${new Date(h.date).toLocaleString('id-ID')}<br>Skor tertinggi: ${escapeHtml(h.top_label)} · ${Number(h.top_probability).toFixed(2)}%</small></div><span class="badge ${h.state}">${h.state==='normal'?'NORMAL':h.state==='perlu-evaluasi'?'EVALUASI':'TIDAK PASTI'}</span></article>`).join(''):'<div class="empty-list">Belum ada hasil screening yang tersimpan di browser ini.</div>';
}

// Simpan hasil setiap kali render inference selesai.
const originalRender = render;
render = function(data){ originalRender(data); saveHistory(data); };
