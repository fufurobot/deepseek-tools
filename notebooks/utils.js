function getParentChain(element) {
  const chain = [];
  let current = element;
  
  while (current) {
    chain.push(current);
    if (current.tagName === 'BODY' || current.tagName === 'HTML') {
      break;
    }
    current = current.parentElement;
  }
  
  return chain;
}

const idMap = {};
let nextId = 0;

function getObjectId(obj) {
  if (!idMap.has(obj)) {
    idMap.set(obj, ++nextId);
  }
  return idMap.get(obj);
}

function getTextWithIndex(node) {
    const text_map = {};
    for (const [index, child] of [...node.childNodes].entries()) {
        if (node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '') {
            text_map[index] = node.textContent.trim();
        }
    }
    return text_map;
}

function DOMNodeAsDict(node) {
  // node.attributes is a NamedNodeMap; convert to an object
  const props = {};
  for (let attr of node.attributes) {
    props[attr.name] = attr.value;
  }
  props["__object_id"] = getObjectId(node);
  props["__tag"] = node.name;
  props["__texts"] = getTextWithIndex(node);
  return props;
}


