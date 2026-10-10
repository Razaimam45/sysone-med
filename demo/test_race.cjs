const assert = require('node:assert/strict');
const {completionTimes, completedAt} = require('./static/race.js');
const data = {total:1000,unique:1,samples:[{id:'a'}],recorded:{
  fast:{a:{latency_ms:10}},slow:{a:{latency_ms:1000}}
}};
const fast = completionTimes(data,'fast'), slow = completionTimes(data,'slow');
assert.equal(completedAt(fast,0),0);
assert.equal(completedAt(fast,1000),100);
assert.equal(completedAt(slow,1000),1);
assert.equal(completedAt(fast,999),99);
assert.equal(completedAt(slow,999),0);
assert.equal(completedAt(fast,1e9),1000);
assert.equal(completedAt(slow,1e9),1000);
console.log('Race timing passed: 100× faster model completes 100 vs 1 at one second.');
