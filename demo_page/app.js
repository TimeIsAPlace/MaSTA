const systems=[['reference','Ground Truth','#7c8c80'],['cosyvoice','CosyVoice2','#4b7e9d'],['maskgct','MaskGCT','#3b8965']];
const list=document.getElementById('sample-list');
function annotated(node,text){
  const regex=/\[[^\]]*\]|\/[bipr]/gi;let from=0;
  for(const match of text.matchAll(regex)){
    node.append(document.createTextNode(text.slice(from,match.index)));
    const span=document.createElement('span');span.className='event';span.textContent=match[0];node.append(span);from=match.index+match[0].length;
  }
  node.append(document.createTextNode(text.slice(from)));
}
function renderDataset(dataset){
document.querySelectorAll('audio').forEach(audio=>audio.pause());
list.replaceChildren();
const samples=window.DEMO_DATA.filter(sample=>sample.dataset===dataset);
document.getElementById('count').textContent=`${samples.length} 组样本`;
document.getElementById('reference-label').textContent=dataset==='aishell1'?'Original Speech':'Ground Truth';
document.getElementById('reference-description').textContent=dataset==='aishell1'?'原始流利语音':'原始口吃语音';
document.querySelectorAll('#filters button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dataset===dataset)));
samples.forEach((sample,index)=>{
  const row=document.createElement('article');row.className='sample-row';
  const text=document.createElement('div');
  const meta=document.createElement('div');meta.className='sample-meta';
  const number=document.createElement('strong');number.textContent=String(index+1).padStart(2,'0');
  meta.append(number);
  const sentence=document.createElement('p');sentence.className='sample-text';annotated(sentence,sample.text);text.append(meta,sentence);row.append(text);
  for(const [key,label,color] of systems){
    const player=document.createElement('div');player.className='player';
    const displayLabel=key==='reference'&&dataset==='aishell1'?'Original Speech':label;
    const heading=document.createElement('span');heading.className='player-label';heading.textContent=displayLabel;
    const canvas=document.createElement('canvas');canvas.className='waveform';canvas.setAttribute('aria-hidden','true');
    const audio=document.createElement('audio');audio.controls=true;audio.preload='metadata';audio.src=sample.audio[key].src;audio.setAttribute('aria-label',`${sample.id} ${displayLabel}：${sample.text}`);
    audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()}));
    audio.addEventListener('error',()=>{if(player.querySelector('.audio-error'))return;const error=document.createElement('span');error.className='audio-error';error.textContent='音频加载失败';player.append(error)});
    player.append(heading,canvas,audio);row.append(player);
    // Peak envelopes are generated from the actual audio during the build.
    const peaks=sample.audio[key].peaks || [];canvas.width=600;canvas.height=76;
    const ctx=canvas.getContext('2d');ctx.fillStyle=color;
    peaks.forEach((value,i)=>{const h=Math.max(2,value*66);ctx.fillRect(i*600/peaks.length,38-h/2,Math.max(1,600/peaks.length-2),h)});
  }
  list.append(row);
});
}
document.querySelectorAll('#filters button').forEach(button=>button.addEventListener('click',()=>renderDataset(button.dataset.dataset)));
renderDataset('as70');
