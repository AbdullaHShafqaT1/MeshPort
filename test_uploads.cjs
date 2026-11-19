// Exercise the browser's actual upload handlers without WebGL or a cloud GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function element() {
  return { children: [], value: '', classList: { toggle() {}, add() {}, remove() {} },
    append(...items) { this.children.push(...items); },
    replaceChildren() { this.children = []; }, setAttribute() {} };
}
const elements = new Map();
const get = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
let created = 0, revoked = 0;
const context = vm.createContext({
  document: { getElementById: get, createElement: element,
    querySelectorAll: () => get('thumbnails').children.map(card => card.children[2]) },
  clearTimeout, setTimeout, URL: { createObjectURL: () => 'blob:' + (++created), revokeObjectURL: () => revoked++ }
});
const source = fs.readFileSync(__dirname + '/app/static/studio.js', 'utf8');
vm.runInContext(fs.readFileSync(__dirname + '/app/static/reference-views.js', 'utf8'), context);
vm.runInContext(source.slice(0, source.indexOf('const encode=')), context);
const file = name => ({ name, type: 'image/png', size: 100 });
const select = files => get('images').onchange({ target: Object.assign(get('images'), { files }) });
select([file('right.png')]);
select([file('left.png'), file('front.png')]);
assert.equal(get('thumbnails').children.length, 3, 'Separate selections accumulate');
assert.match(get('viewSummary').textContent, /Multi-view/);
get('thumbnails').children[1].children[2].onclick();
assert.equal(get('thumbnails').children.length, 2);
assert.equal(get('thumbnails').children[1].children[1].textContent, 'front.png');
select(Array.from({ length: 5 }, (_, i) => file(i + '.png')));
assert.equal(get('thumbnails').children.length, 2, 'Count overflow preserves selection');
select([{ name: 'large.png', type: 'image/png', size: 24 * 1024 * 1024 }]);
assert.equal(get('thumbnails').children.length, 2, 'Total byte limit includes earlier views');
select([{ name: 'invalid.txt', type: 'text/plain', size: 10 }]);
assert.equal(get('thumbnails').children.length, 2, 'Invalid format preserves selection');
vm.runInContext('running=true;ready()', context);
assert.ok(get('thumbnails').children.every(card => card.children[2].disabled));
get('thumbnails').children[0].children[2].onclick();
assert.equal(get('thumbnails').children.length, 2, 'Cannot remove during generation');
vm.runInContext('running=false', context);
get('clear').onclick();
assert.equal(get('thumbnails').children.length, 0);
assert.equal(revoked, created, 'All thumbnail object URLs released');
console.log('Upload accumulation, removal, limits, busy state and cleanup passed.');
