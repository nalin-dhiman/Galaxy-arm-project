/* Pure numerical helpers. No dependencies, network requests or hidden models. */
export const ARCSEC_PER_RADIAN=206264.80624709636;
export const G=4.30091e-6; // kpc (km/s)^2 / solar mass
function positive(value,name){if(!Number.isFinite(value)||value<=0)throw new RangeError(`${name} must be positive and finite`);return value;}
function point(value,name='Image point'){if(!Array.isArray(value)||value.length!==2||!value.every(Number.isFinite))throw new RangeError(`${name} must contain two finite coordinates`);return value;}
function diskGeometry(center,inclinationDeg,positionAngleDeg){point(center,'Center');if(!Number.isFinite(inclinationDeg)||inclinationDeg<0||inclinationDeg>80)throw new RangeError('Disk inclination must be between 0 and 80 degrees');if(!Number.isFinite(positionAngleDeg))throw new RangeError('Disk angle must be finite');const a=positionAngleDeg*Math.PI/180;return {c:Math.cos(a),s:Math.sin(a),q:Math.cos(inclinationDeg*Math.PI/180)};}
// Display coordinates: x increases right, y down. Disk angle is counterclockwise
// from display +x in a y-up plane, NOT a celestial PA without an external WCS.
export function deprojectPoint(imagePoint,center,inclinationDeg=0,positionAngleDeg=0){point(imagePoint);const {c,s,q}=diskGeometry(center,inclinationDeg,positionAngleDeg),x=imagePoint[0]-center[0],y=center[1]-imagePoint[1];return [x*c+y*s,(-x*s+y*c)/q];}
export function projectPoint(diskPoint,center,inclinationDeg=0,positionAngleDeg=0){point(diskPoint,'Disk point');const {c,s,q}=diskGeometry(center,inclinationDeg,positionAngleDeg),x=diskPoint[0],y=diskPoint[1]*q;return [center[0]+x*c-y*s,center[1]-x*s-y*c];}
export function fitLogSpiral(imagePoints,center,inclinationDeg=0,positionAngleDeg=0){
  if(!Array.isArray(imagePoints)||imagePoints.length<3)throw new RangeError('Mark at least three ordered points along one arm');
  const points=imagePoints.map(p=>deprojectPoint(p,center,inclinationDeg,positionAngleDeg)),radii=points.map(p=>Math.hypot(...p));
  if(radii.some(r=>r<1e-6))throw new RangeError('An arm point cannot coincide with the disk center');
  const angles=[];for(const p of points){let a=Math.atan2(p[1],p[0]);if(angles.length){while(a-angles.at(-1)>Math.PI)a-=2*Math.PI;while(a-angles.at(-1)<-Math.PI)a+=2*Math.PI;}angles.push(a);}
  const differences=angles.slice(1).map((a,i)=>a-angles[i]);if(differences.some(d=>Math.abs(d)>Math.PI*.95))throw new RangeError('Neighboring arm points are nearly 180 degrees apart; add intermediate points');
  const direction=Math.sign(angles.at(-1)-angles[0]);if(!direction||differences.some(d=>d*direction<-.01))throw new RangeError('Trace one arm in angular order without reversing direction');
  const span=Math.max(...angles)-Math.min(...angles);if(span<20*Math.PI/180)throw new RangeError('Trace at least 20 degrees of azimuth for a pitch fit');
  const logs=radii.map(Math.log),n=angles.length,meanAngle=angles.reduce((a,b)=>a+b,0)/n,meanLog=logs.reduce((a,b)=>a+b,0)/n;
  const xx=angles.reduce((s,a)=>s+(a-meanAngle)**2,0),xy=angles.reduce((s,a,i)=>s+(a-meanAngle)*(logs[i]-meanLog),0),slope=xy/xx,intercept=meanLog-slope*meanAngle;
  const residuals=angles.map((a,i)=>logs[i]-(intercept+slope*a)),sse=residuals.reduce((s,r)=>s+r*r,0),tss=logs.reduce((s,r)=>s+(r-meanLog)**2,0);
  return {count:n,slope,intercept,signedPitchDeg:Math.atan(slope)*180/Math.PI,absolutePitchDeg:Math.abs(Math.atan(slope)*180/Math.PI),angleStart:Math.min(...angles),angleEnd:Math.max(...angles),azimuthSpanDeg:span*180/Math.PI,rmsLogRadius:Math.sqrt(sse/n),rSquared:tss>1e-20?1-sse/tss:null,radiiPixels:radii,anglesRadians:angles,residualLogRadius:residuals};
}
export function radialSpacing(imagePoints,center,inclinationDeg=0,positionAngleDeg=0,pixelScale=null,distanceMpc=null){
  if(!Array.isArray(imagePoints)||imagePoints.length<2)throw new RangeError('Mark at least two arm intersections on the same radial ray');
  if(pixelScale!==null)positive(pixelScale,'Original pixel scale');if(distanceMpc!==null)positive(distanceMpc,'Angular diameter distance');
  const p=imagePoints.map(v=>deprojectPoint(v,center,inclinationDeg,positionAngleDeg)),r=p.map(v=>Math.hypot(...v));if(r.some(v=>v<1e-6))throw new RangeError('A radial mark cannot coincide with the disk center');
  const angles=p.map(v=>Math.atan2(v[1],v[0])),mean=Math.atan2(angles.reduce((s,a)=>s+Math.sin(a),0),angles.reduce((s,a)=>s+Math.cos(a),0));
  const spread=Math.max(...angles.map(a=>Math.abs(Math.atan2(Math.sin(a-mean),Math.cos(a-mean)))))*180/Math.PI;
  if(spread>12)throw new RangeError('Radial marks must lie on one ray within 12 degrees; opposite sides are different rays');
  const radii=[...r].sort((a,b)=>a-b),gaps=radii.slice(1).map((v,i)=>v-radii[i]);if(gaps.some(v=>v<1e-6))throw new RangeError('Radial marks must have distinct radii');
  return {count:r.length,radiiPixels:radii,gapsPixels:gaps,meanGapPixels:gaps.reduce((a,b)=>a+b,0)/gaps.length,rayAngleDeg:mean*180/Math.PI,maxRayDeviationDeg:spread,gapsArcsec:pixelScale===null?null:gaps.map(v=>v*pixelScale),gapsKpc:pixelScale===null||distanceMpc===null?null:gaps.map(v=>v*pixelScale*distanceMpc*1000/ARCSEC_PER_RADIAN)};
}
export function separation(a,b,pixelScale=null,distanceMpc=null){
  point(a,'First image point');point(b,'Second image point');
  const pixels=Math.hypot(a[0]-b[0],a[1]-b[1]);
  const arcsec=pixelScale===null?null:pixels*positive(pixelScale,'Original pixel scale');
  const kpc=distanceMpc===null?null:(positive(distanceMpc,'Angular diameter distance')*1000*(arcsec===null?NaN:arcsec)/ARCSEC_PER_RADIAN);
  return {pixels,arcsec,kpc:Number.isNaN(kpc)?null:kpc};
}
export function displayChannel(rgba,channel='gray'){
  const out=new Float32Array(rgba.length/4);let weights=channel==='red'?[1,0,0]:channel==='green'?[0,1,0]:channel==='blue'?[0,0,1]:[.2126,.7152,.0722];
  for(let i=0;i<out.length;i++)out[i]=weights[0]*rgba[4*i]+weights[1]*rgba[4*i+1]+weights[2]*rgba[4*i+2];
  return out; // display luma or encoded display channel; NOT calibrated flux
}
export function gaussianBlur(data,width,height,sigma=1.4){
  if(data.length!==width*height)throw new RangeError('Image shape mismatch');positive(sigma,'Filter sigma');
  const radius=Math.ceil(3*sigma),kernel=[];let sum=0;
  for(let k=-radius;k<=radius;k++){let v=Math.exp(-.5*(k/sigma)**2);kernel.push(v);sum+=v;}
  for(let k=0;k<kernel.length;k++)kernel[k]/=sum;
  const temp=new Float32Array(data.length),out=new Float32Array(data.length);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let k=-radius;k<=radius;k++)temp[y*width+x]+=data[y*width+Math.max(0,Math.min(width-1,x+k))]*kernel[k+radius];
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let k=-radius;k<=radius;k++)out[y*width+x]+=temp[Math.max(0,Math.min(height-1,y+k))*width+x]*kernel[k+radius];
  return out;
}
export function filterPixels(rgba,width,height,mode='gray',sigma=1.4){
  if(rgba.length!==width*height*4)throw new RangeError('RGBA shape mismatch');
  if(mode==='original')return new Uint8ClampedArray(rgba);
  const gray=displayChannel(rgba,['red','green','blue'].includes(mode)?mode:'gray');let score=gray;
  if(['smooth','unsharp','gradient','ridge'].includes(mode)){
    const smooth=gaussianBlur(gray,width,height,sigma);score=new Float32Array(gray.length);
    if(mode==='smooth')score=smooth;
    else if(mode==='unsharp')for(let i=0;i<gray.length;i++)score[i]=gray[i]+1.8*(gray[i]-smooth[i]);
    else for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      let i=y*width+x,xm=Math.max(0,x-1),xp=Math.min(width-1,x+1),ym=Math.max(0,y-1),yp=Math.min(height-1,y+1);
      if(mode==='gradient')score[i]=Math.hypot((smooth[y*width+xp]-smooth[y*width+xm])/2,(smooth[yp*width+x]-smooth[ym*width+x])/2);
      else {let xx=smooth[y*width+xp]+smooth[y*width+xm]-2*smooth[i],yy=smooth[yp*width+x]+smooth[ym*width+x]-2*smooth[i];let xy=(smooth[yp*width+xp]-smooth[yp*width+xm]-smooth[ym*width+xp]+smooth[ym*width+xm])/4;score[i]=Math.max(0,-.5*(xx+yy-Math.sqrt((xx-yy)**2+4*xy**2)));}
    }
    if(mode==='gradient'||mode==='ridge'){const sample=[];for(let i=0;i<score.length;i+=Math.max(1,Math.floor(score.length/6000)))sample.push(score[i]);sample.sort((a,b)=>a-b);const ceiling=sample[Math.floor(.99*(sample.length-1))]||1;for(let i=0;i<score.length;i++)score[i]=255*score[i]/ceiling;}
  }
  const result=new Uint8ClampedArray(rgba.length);for(let i=0;i<score.length;i++){const v=Math.max(0,Math.min(255,score[i]));result[4*i]=result[4*i+1]=result[4*i+2]=v;result[4*i+3]=255;}return result;
}
export function kingMoment(W,pressure=false){
  if(!Number.isFinite(W)||W>100)throw new RangeError('Finite binding W≤100 required');if(W<=0)return 0;
  // Positive series avoids subtractive cancellation near the finite edge.
  let power=pressure?3.5:2.5,term=(pressure?16/105:8/15)/Math.sqrt(Math.PI)*W**power,total=term;
  for(let i=0;i<180;i++){term*=W/(power+1);total+=term;power++;if(term<Math.abs(total)*1e-15)break;}return total;
}
function rhs(x,W,m,normal){return [-m/(x*x),x*x*kingMoment(W)/normal];}
function rk4(x,W,m,h,normal){
  let a=rhs(x,W,m,normal),b=rhs(x+h/2,W+h*a[0]/2,m+h*a[1]/2,normal),c=rhs(x+h/2,W+h*b[0]/2,m+h*b[1]/2,normal),d=rhs(x+h,W+h*c[0],m+h*c[1],normal);
  return [W+h*(a[0]+2*b[0]+2*c[0]+d[0])/6,m+h*(a[1]+2*b[1]+2*c[1]+d[1])/6];
}
export function sphericalHalo(rho0=1e7,velocityScale=100,W0=5){
  positive(rho0,'Central density');positive(velocityScale,'Velocity scale');
  if(rho0<1e5||rho0>1e10||velocityScale<5||velocityScale>500||!Number.isFinite(W0)||W0<1||W0>8)throw new RangeError('Browser bounds: 1≤W₀≤8, 10⁵≤ρ₀≤10¹⁰, 5≤s≤500');
  const normal=kingMoment(W0),scale=velocityScale/Math.sqrt(4*Math.PI*G*rho0);let x=1e-4,W=W0-x*x/6,m=x*x*x/3;
  const states=[[0,W0,0],[x,W,m]];let edge=null;
  for(let step=0;step<30000;step++){
    const h=Math.min(.15,.025*x+.001),next=rk4(x,W,m,h,normal);
    if(next[0]<=0){const f=W/(W-next[0]);edge=x+h*f;m=m+(next[1]-m)*f;states.push([edge,0,m]);break;}
    x+=h;[W,m]=next;states.push([x,W,m]);
  }
  if(edge===null)throw new Error('Finite edge was not resolved');
  const totalMass=4*Math.PI*rho0*scale**3*m;
  const rows=states.map(([x,W,m])=>({x,radiusKpc:x*scale,W,m,density:rho0*kingMoment(W)/normal,pressure:rho0*velocityScale**2*kingMoment(W,true)/normal,velocity:x>0?velocityScale*Math.sqrt(Math.max(0,m/x)):0}));
  for(let j=1;j<=30;j++){let x=edge*(1+.25*j/30);rows.push({x,radiusKpc:x*scale,W:m*(1/x-1/edge),m,density:0,pressure:0,velocity:velocityScale*Math.sqrt(m/x)});}
  return {rows,scaleKpc:scale,edgeKpc:edge*scale,edgeX:edge,totalMass,rho0,velocityScale,W0,method:'spherical lowered-isothermal equilibrium; no baryonic disk; no viscosity inferred'};
}
export function cohortSummary(records,cohort,consensus=3){
  const rows=records.filter(r=>(cohort==='all'||r.cohort===cohort)&&r.consensus===consensus),valid=rows.filter(r=>Number.isFinite(r.ap));
  return {predicted:rows.length,evaluable:valid.length,meanAP:valid.length?valid.reduce((s,r)=>s+r.ap,0)/valid.length:null,meanF1:valid.length?valid.reduce((s,r)=>s+r.f1,0)/valid.length:null};
}
