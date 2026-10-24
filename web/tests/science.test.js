import test from 'node:test';import assert from 'node:assert/strict';
import {separation,kingMoment,sphericalHalo,gaussianBlur,filterPixels,cohortSummary,deprojectPoint,projectPoint,fitLogSpiral,radialSpacing} from '../dist/science.js';
const near=(a,b,t)=>assert.ok(Math.abs(a-b)<=t,`${a} differs from ${b} by >${t}`);
test('Angular calibration and external D_A conversion, including missing and invalid calibration',()=>{let r=separation([0,0],[3,4],.4,10);near(r.pixels,5,1e-14);near(r.arcsec,2,1e-14);near(r.kpc,.0969627362219,1e-12);assert.equal(separation([0,0],[1,1]).arcsec,null);assert.equal(separation([0,0],[1,1],null,10).kpc,null);assert.throws(()=>separation([0,0],[1,1],-1),RangeError);assert.throws(()=>separation([NaN,0],[1,1],1),RangeError);});
test('Display smoothing preserves uniform input and symmetric point source',()=>{let x=new Float32Array(25*25).fill(42);for(let v of gaussianBlur(x,25,25,1.4))near(v,42,2e-5);x.fill(0);x[12*25+12]=1;let y=gaussianBlur(x,25,25,1.4);near(y[12*25+10],y[12*25+14],1e-8);near(y.reduce((a,b)=>a+b),1,2e-7);assert.throws(()=>gaussianBlur(x,4,5,1));});
test('Display edges and ridges do not create structure in a uniform field',()=>{let rgba=new Uint8ClampedArray(30*30*4);for(let i=0;i<900;i++){rgba[4*i]=rgba[4*i+1]=rgba[4*i+2]=100;rgba[4*i+3]=255;}for(const mode of ['gradient','ridge']){let result=filterPixels(rgba,30,30,mode);for(let i=0;i<900;i++)assert.equal(result[4*i],0);}assert.deepEqual(filterPixels(rgba,30,30,'original'),rgba);});
test('King pressure derivative is the density moment and finite edge is exact',()=>{for(let W of [.001,.1,1,5,8]){let h=1e-5*Math.max(W,.01),derivative=(kingMoment(W+h,true)-kingMoment(W-h,true))/(2*h);near(derivative/kingMoment(W),1,1e-6);}assert.equal(kingMoment(-1),0);assert.equal(kingMoment(0,true),0);});
test('Browser King ODE agrees with independently integrated Python reference',()=>{for(const [W,edge,mass] of [[1,25.482561825007313,21598743273.972492],[3,60.64274867842585,111420428080.17685],[5,138.0383584020158,253933975706.68872],[7,434.98683343140294,535924181063.4246]]){let h=sphericalHalo(1e7,100,W);near(h.edgeKpc/edge,1,2e-4);near(h.totalMass/mass,1,1e-5);assert.equal(h.rows[0].velocity,0);near(h.rows[0].density,1e7,1e-7);assert.equal(h.rows.at(-1).density,0);assert.equal(h.rows.at(-1).pressure,0);let edgeRow=h.rows.find(r=>r.W===0);assert.ok(edgeRow);near(h.rows.at(-1).velocity/edgeRow.velocity,1/Math.sqrt(1.25),1e-12);for(let i=1;i<h.rows.length;i++)assert.ok(h.rows[i].m>=h.rows[i-1].m-1e-10);}});
test('Halo physical scaling and hydrostatic balance are consistent',()=>{let a=sphericalHalo(1e7,100,5),b=sphericalHalo(4e7,100,5),c=sphericalHalo(1e7,200,5);near(b.scaleKpc/a.scaleKpc,.5,1e-12);near(b.totalMass/a.totalMass,.5,1e-12);near(c.scaleKpc/a.scaleKpc,2,1e-12);near(c.totalMass/a.totalMass,8,1e-12);let errors=[];for(let i=8;i<a.rows.length-32;i++){let prev=a.rows[i-1],r=a.rows[i],next=a.rows[i+1];if(r.W<.2||r.x<.3)continue;let dp=(next.pressure-prev.pressure)/(next.radiusKpc-prev.radiusKpc),mass=4*Math.PI*a.rho0*a.scaleKpc**3*r.m,expected=-r.density*4.30091e-6*mass/(r.radiusKpc**2);errors.push(Math.abs(dp/expected-1));}assert.ok(errors.reduce((a,b)=>a+b)/errors.length<.005);assert.throws(()=>sphericalHalo(1e7,100,12),RangeError);});
test('Unknown annotation truth stays undefined, rather than becoming a negative',()=>{let r=cohortSummary([{cohort:'unknown',consensus:3,ap:null,f1:null}], 'unknown');assert.equal(r.evaluable,0);assert.equal(r.meanAP,null);let known=cohortSummary([{cohort:'spiral',consensus:3,ap:.5,f1:.4},{cohort:'spiral',consensus:3,ap:.7,f1:.6}], 'spiral');near(known.meanAP,.6,1e-15);});
test('Display axes and disk angle use y-up mathematical azimuth, with invertible thin-disk projection',()=>{
  assert.deepEqual(deprojectPoint([110,100],[100,100]),[10,0]);near(deprojectPoint([100,90],[100,100])[1],10,1e-14);
  const major=projectPoint([10,0],[100,100],60,90);near(major[0],100,1e-12);near(major[1],90,1e-12);
  const minor=projectPoint([0,10],[100,100],60,0);near(minor[0],100,1e-12);near(minor[1],95,1e-12);
  for(const inclination of [0,35,80])for(const angle of [-170,0,38,280]){const p=projectPoint([31,-14],[103,97],inclination,angle),q=deprojectPoint(p,[103,97],inclination,angle);near(q[0],31,1e-12);near(q[1],-14,1e-12);}
  assert.throws(()=>deprojectPoint([1,2],[0,0],81),RangeError);assert.throws(()=>projectPoint([NaN,2],[0,0]),RangeError);
});
test('Manual pitch recovers a known projected logarithmic arm through an angle wrap in either trace direction',()=>{
  for(const pitch of [-24,17,0]){const slope=Math.tan(pitch*Math.PI/180),points=Array.from({length:21},(_,i)=>{const a=2.4+i*.11,r=20*Math.exp(slope*(a-2.4));return projectPoint([r*Math.cos(a),r*Math.sin(a)],[180,210],57,37);});
    for(const trace of [points,[...points].reverse()]){const fit=fitLogSpiral(trace,[180,210],57,37);near(fit.signedPitchDeg,pitch,1e-11);near(fit.rmsLogRadius,0,1e-13);near(fit.azimuthSpanDeg,2.2*180/Math.PI,1e-11);if(pitch===0)assert.equal(fit.rSquared,null);else near(fit.rSquared,1,1e-13);}
  }
});
test('Manual pitch rejects center, too short, reversed and near half-turn traces rather than supplying unstable geometry',()=>{
  const p=a=>[100+20*Math.cos(a),100-20*Math.sin(a)];
  assert.throws(()=>fitLogSpiral([p(0),p(.2)],[100,100]),RangeError);
  assert.throws(()=>fitLogSpiral([p(0),p(.05),p(.1)],[100,100]),/20 degrees/);
  assert.throws(()=>fitLogSpiral([[100,100],p(.4),p(.8)],[100,100]),/center/);
  assert.throws(()=>fitLogSpiral([p(0),p(.8),p(.4)],[100,100]),/angular order/);
  assert.throws(()=>fitLogSpiral([p(0),p(3.1),p(3.3)],[100,100]),/180 degrees/);
});
test('Manual radial gaps sort same-ray disk intersections and require external angular and distance calibration',()=>{
  const points=[40,10,25].map(r=>projectPoint([r*Math.cos(3.1),r*Math.sin(3.1)],[100,100],62,17));
  const r=radialSpacing(points,[100,100],62,17,.4,10);r.radiiPixels.forEach((v,i)=>near(v,[10,25,40][i],1e-12));r.gapsPixels.forEach(v=>near(v,15,1e-12));r.gapsArcsec.forEach(v=>near(v,6,1e-12));r.gapsKpc.forEach(v=>near(v,.2908882086657,1e-12));near(r.maxRayDeviationDeg,0,1e-11);
  assert.equal(radialSpacing(points,[100,100],62,17,null,10).gapsKpc,null);assert.equal(radialSpacing(points,[100,100],62,17,.4).gapsKpc,null);
  assert.throws(()=>radialSpacing(points,[100,100],62,17,-1),RangeError);assert.throws(()=>radialSpacing(points,[100,100],62,17,.4,-2),RangeError);
  assert.throws(()=>radialSpacing([[110,100],[90,100]],[100,100]),/one ray/);assert.throws(()=>radialSpacing([[110,100],[110,100]],[100,100]),/distinct/);assert.throws(()=>radialSpacing([[100,100],[110,100]],[100,100]),/center/);
});
