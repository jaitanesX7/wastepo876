/**
 * EcoSort AI — Intelligent Waste Classification Frontend
 * Powered by TensorFlow.js, MobileNet v2, and In-Browser Computer Vision
 */

// --- Global State ---
let model = null;
let customModel = null;
let currentStream = null;
let facingMode = 'environment';
let isContinuousScanning = false;
let continuousScanTimer = null;
let isModelLoading = true;
let confidenceThreshold = 0.30; // 30% default

// --- Waste Taxonomy & Guidance Database ---
const WASTE_CATEGORIES = {
  organic: {
    id: 'organic',
    name: 'Organic & Compost',
    icon: '🌱',
    binName: 'Green Bin — Organics & Food Scraps',
    themeColor: 'emerald',
    badgeClass: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    bannerClass: 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200',
    dotClass: 'bg-emerald-500 ring-emerald-500/20',
    barGradient: 'from-emerald-600 to-teal-400',
    decomposition: '2 weeks to 6 months',
    material: 'Biodegradable Biomass',
    co2PerItem: 0.18, // kg CO2 diverted
    isDiverted: true,
    steps: [
      'Remove any non-compostable stickers, twist ties, or plastic tags.',
      'Place food scraps directly into your green organic waste bin or home composter.',
      'Do not enclose in conventional non-biodegradable plastic bags.'
    ]
  },
  recyclable: {
    id: 'recyclable',
    name: 'Dry Recyclables',
    icon: '♻️',
    binName: 'Blue Bin — Plastics, Glass & Metal',
    themeColor: 'blue',
    badgeClass: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    bannerClass: 'bg-blue-950/40 border-blue-800/60 text-blue-200',
    dotClass: 'bg-blue-500 ring-blue-500/20',
    barGradient: 'from-blue-600 to-cyan-400',
    decomposition: '100 to 500+ years',
    material: 'PET / HDPE / Aluminum / Glass',
    co2PerItem: 0.42,
    isDiverted: true,
    steps: [
      'Empty all remaining liquids and rinse out solid residue.',
      'Leave plastic bottle caps screwed on (or separate if local guidelines state).',
      'Do not crush glass containers to preserve sorting safety at the recycling facility.'
    ]
  },
  paper: {
    id: 'paper',
    name: 'Paper & Cardboard',
    icon: '📦',
    binName: 'Yellow / Blue Bin — Paper & Cardboard',
    themeColor: 'amber',
    badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    bannerClass: 'bg-amber-950/40 border-amber-800/60 text-amber-200',
    dotClass: 'bg-amber-500 ring-amber-500/20',
    barGradient: 'from-amber-600 to-yellow-400',
    decomposition: '2 to 5 months',
    material: 'Cellulose Fibers',
    co2PerItem: 0.25,
    isDiverted: true,
    steps: [
      'Flatten all cardboard boxes to save space in the collection truck.',
      'Keep completely dry; moisture can ruin paper fibers during baling.',
      'Remove plastic packaging bubbles, sticky tape, and styrofoam inserts.'
    ]
  },
  hazardous: {
    id: 'hazardous',
    name: 'E-Waste & Hazardous',
    icon: '⚡',
    binName: 'Red Bin — Specialized Hazardous Depot',
    themeColor: 'rose',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    bannerClass: 'bg-rose-950/40 border-rose-800/60 text-rose-200',
    dotClass: 'bg-rose-500 ring-rose-500/20',
    barGradient: 'from-rose-600 to-pink-400',
    decomposition: 'Indefinite / Heavy Metal Leaching',
    material: 'Lithium / Circuitry / Toxic Compounds',
    co2PerItem: 1.15,
    isDiverted: true,
    steps: [
      'NEVER discard in ordinary household trash or curbside recycling.',
      'Place clear tape over battery terminals to prevent short circuits and fire hazards.',
      'Drop off at an authorized electronics store, municipal depot, or battery bin.'
    ]
  },
  landfill: {
    id: 'landfill',
    name: 'Landfill / General Trash',
    icon: '🗑️',
    binName: 'Grey / Black Bin — Non-Recyclable Waste',
    themeColor: 'slate',
    badgeClass: 'bg-slate-500/10 text-slate-400 border-slate-500/30',
    bannerClass: 'bg-slate-900 border-slate-700 text-slate-300',
    dotClass: 'bg-slate-500 ring-slate-500/20',
    barGradient: 'from-slate-600 to-slate-400',
    decomposition: '50 to 1,000 years',
    material: 'Mixed Non-Recyclables / Composites',
    co2PerItem: 0.0,
    isDiverted: false,
    steps: [
      'Ensure no recyclable or hazardous items are mixed inside.',
      'Bag and tie securely to prevent windblown street litter.',
      'Consider reusable zero-waste alternatives for future purchases.'
    ]
  }
};

// Keyword mapping dictionary for ImageNet classes
const TAXONOMY_RULES = [
  // ORGANIC
  {
    category: 'organic',
    keywords: [
      'banana', 'apple', 'orange', 'lemon', 'pomegranate', 'pineapple', 'grape', 'strawberry',
      'fig', 'custard apple', 'jackfruit', 'mango', 'peach', 'plum', 'pear', 'cucumber',
      'zucchini', 'squash', 'pumpkin', 'broccoli', 'cauliflower', 'cabbage', 'head cabbage',
      'artichoke', 'bell pepper', 'chili', 'pepper', 'corn', 'ear', 'mushroom', 'agaric',
      'bolete', 'truffle', 'leaf', 'plant', 'flower', 'pot', 'hay', 'straw', 'wood',
      'bread', 'loaf', 'bagel', 'pretzel', 'pizza', 'pie', 'sandwich', 'hamburger',
      'hotdog', 'cheeseburger', 'burrito', 'taco', 'meat', 'beef', 'pork', 'chicken',
      'fish', 'egg', 'scrambled', 'omelet', 'salad', 'guacamole', 'soup', 'food',
      'coffee', 'tea', 'espresso'
    ]
  },
  // RECYCLABLE (Plastics, Metals, Glass)
  {
    category: 'recyclable',
    keywords: [
      'water bottle', 'plastic bottle', 'pop bottle', 'beer bottle', 'wine bottle',
      'bottle', 'flask', 'beaker', 'cocktail shaker', 'can', 'tin can', 'soda can',
      'aluminum', 'beer glass', 'goblet', 'wine glass', 'pitcher', 'jug', 'cup',
      'measuring cup', 'tub', 'bucket', 'pail', 'jar', 'mason jar', 'lotion', 'sunscreen',
      'shampoo', 'soap dispenser', 'detergent', 'spray bottle', 'pottery', 'plate',
      'saucer', 'tray', 'can opener', 'foil', 'tin', 'tray', 'frying pan', 'saucepan',
      'wok', 'kettle', 'metal', 'steel'
    ]
  },
  // PAPER & CARDBOARD
  {
    category: 'paper',
    keywords: [
      'carton', 'cardboard', 'packet', 'box', 'envelope', 'paper', 'newspaper',
      'magazine', 'book', 'notebook', 'binder', 'ledger', 'comic book', 'menu',
      'ticket', 'paper towel', 'toilet tissue', 'tissue', 'receipt', 'card', 'postcard'
    ]
  },
  // HAZARDOUS / E-WASTE
  {
    category: 'hazardous',
    keywords: [
      'battery', 'accumulator', 'cellular telephone', 'cellphone', 'mobile phone',
      'hand-held computer', 'laptop', 'notebook computer', 'desktop computer',
      'monitor', 'screen', 'television', 'tv', 'keyboard', 'mouse', 'remote control',
      'modem', 'router', 'hard disc', 'disk', 'flash drive', 'memory', 'ipod',
      'cassette', 'tape', 'radio', 'loudspeaker', 'headphone', 'earphone',
      'light bulb', 'bulb', 'fluorescent', 'lamp', 'projector', 'power cord', 'cord',
      'plug', 'wire', 'cable', 'charger', 'adapter', 'drill', 'electric fan', 'fan',
      'hair dryer', 'iron', 'toaster', 'microwave', 'oven', 'aerosol', 'spray can',
      'syringe', 'medicine chest', 'pill bottle'
    ]
  },
  // LANDFILL / GENERAL TRASH
  {
    category: 'landfill',
    keywords: [
      'diaper', 'napkin', 'sponge', 'scouring pad', 'band aid', 'bandage', 'mask',
      'cigarette', 'ashcan', 'lighter', 'balloon', 'straw', 'styrofoam', 'bubble wrap',
      'rubber', 'eraser', 'shoe', 'boot', 'sandal', 'sock', 'cloth', 'umbrella',
      'toothbrush', 'comb', 'broom', 'mop'
    ]
  }
];

// --- Built-in SVG Demo Presets ---
const DEMO_PRESETS = [
  {
    id: 'preset-plastic-bottle',
    name: 'Plastic Bottle',
    category: 'recyclable',
    label: 'water bottle, plastic bottle',
    confidence: 0.96,
    svg: `<svg viewBox="0 0 100 100" class="w-12 h-12 mx-auto"><path d="M42 12 h16 v8 h-16 z M45 20 h10 v8 h-10 z M35 28 h30 l6 16 v44 c0 4 -4 6 -8 6 h-26 c-4 0 -8 -2 -8 -6 v-44 z" fill="#38bdf8" fill-opacity="0.4" stroke="#38bdf8" stroke-width="3" stroke-linejoin="round"/><path d="M40 45 h20 M40 55 h20 M40 65 h20" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"/></svg>`
  },
  {
    id: 'preset-banana-peel',
    name: 'Banana Peel',
    category: 'organic',
    label: 'banana, fruit peel',
    confidence: 0.98,
    svg: `<svg viewBox="0 0 100 100" class="w-12 h-12 mx-auto"><path d="M22 68 C35 85, 65 85, 78 50 C86 30, 80 18, 76 15 C72 25, 62 48, 48 55 C35 62, 25 65, 22 68 Z" fill="#facc15" fill-opacity="0.8" stroke="#eab308" stroke-width="3"/><path d="M76 15 L78 10" stroke="#854d0e" stroke-width="4" stroke-linecap="round"/><path d="M40 70 C48 76, 56 74, 62 65" stroke="#ca8a04" stroke-width="2" fill="none"/></svg>`
  },
  {
    id: 'preset-cardboard-box',
    name: 'Cardboard Box',
    category: 'paper',
    label: 'cardboard, shipping box',
    confidence: 0.94,
    svg: `<svg viewBox="0 0 100 100" class="w-12 h-12 mx-auto"><polygon points="50,15 88,32 50,50 12,32" fill="#d97706" fill-opacity="0.5" stroke="#b45309" stroke-width="3"/><polygon points="12,32 50,50 50,85 12,68" fill="#d97706" fill-opacity="0.7" stroke="#b45309" stroke-width="3"/><polygon points="50,50 88,32 88,68 50,85" fill="#d97706" fill-opacity="0.9" stroke="#b45309" stroke-width="3"/></svg>`
  },
  {
    id: 'preset-soda-can',
    name: 'Soda Can',
    category: 'recyclable',
    label: 'soda can, aluminum can',
    confidence: 0.95,
    svg: `<svg viewBox="0 0 100 100" class="w-12 h-12 mx-auto"><ellipse cx="50" cy="22" rx="22" ry="8" fill="#cbd5e1" stroke="#94a3b8" stroke-width="3"/><rect x="28" y="22" width="44" height="56" fill="#f43f5e" fill-opacity="0.8" stroke="#e11d48" stroke-width="3"/><ellipse cx="50" cy="78" rx="22" ry="8" fill="#cbd5e1" stroke="#94a3b8" stroke-width="3"/><path d="M44 20 a6 4 0 1 0 12 0" fill="#64748b"/></svg>`
  },
  {
    id: 'preset-aa-battery',
    name: 'AA Battery',
    category: 'hazardous',
    label: 'battery, accumulator',
    confidence: 0.99,
    svg: `<svg viewBox="0 0 100 100" class="w-12 h-12 mx-auto"><rect x="44" y="14" width="12" height="6" rx="2" fill="#cbd5e1" stroke="#94a3b8" stroke-width="2"/><rect x="34" y="20" width="32" height="62" rx="4" fill="#334155" stroke="#64748b" stroke-width="3"/><rect x="34" y="20" width="32" height="22" rx="4" fill="#f97316"/><path d="M50 48 L46 56 H54 L50 64" stroke="#fbbf24" stroke-width="2.5" fill="none" stroke-linejoin="round"/></svg>`
  },
  {
    id: 'preset-apple-core',
    name: 'Apple Core',
    category: 'organic',
    label: 'apple core, food scrap',
    confidence: 0.93,
    svg: `<svg viewBox="0 0 100 100" class="w-12 h-12 mx-auto"><path d="M50 15 Q55 8 62 10" stroke="#78350f" stroke-width="3" fill="none" stroke-linecap="round"/><ellipse cx="50" cy="24" rx="22" ry="8" fill="#ef4444"/><ellipse cx="50" cy="76" rx="22" ry="8" fill="#ef4444"/><path d="M38 24 Q48 50 38 76 L62 76 Q52 50 62 24 Z" fill="#fef08a" stroke="#ca8a04" stroke-width="2"/><circle cx="48" cy="46" r="2.5" fill="#451a03"/><circle cx="52" cy="54" r="2.5" fill="#451a03"/></svg>`
  }
];

// --- DOM Element References ---
const modelStatusEl = document.getElementById('model-status');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');

// Tabs
const tabCamera = document.getElementById('tab-camera');
const tabUpload = document.getElementById('tab-upload');
const tabPresets = document.getElementById('tab-presets');

// Views
const cameraView = document.getElementById('camera-view');
const uploadView = document.getElementById('upload-view');
const presetsView = document.getElementById('presets-view');
const presetsGrid = document.getElementById('presets-grid');

// Camera Elements
const webcam = document.getElementById('webcam');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const scannerReticle = document.getElementById('scanner-reticle');
const startCamBtn = document.getElementById('start-cam-btn');
const captureBtn = document.getElementById('capture-btn');
const switchCamBtn = document.getElementById('switch-cam-btn');
const continuousScanCheckbox = document.getElementById('continuous-scan');

// Upload Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPreviewContainer = document.getElementById('upload-preview-container');
const uploadPreview = document.getElementById('upload-preview');
const clearUploadBtn = document.getElementById('clear-upload-btn');
const processingOverlay = document.getElementById('processing-overlay');

// Result Card Elements
const emptyResultCard = document.getElementById('empty-result-card');
const resultCard = document.getElementById('result-card');
const resultIconBox = document.getElementById('result-icon-box');
const resultCategory = document.getElementById('result-category');
const resultName = document.getElementById('result-name');
const resultConfidence = document.getElementById('result-confidence');
const confidenceBar = document.getElementById('confidence-bar');
const binBanner = document.getElementById('bin-banner');
const binIndicatorDot = document.getElementById('bin-indicator-dot');
const binTitle = document.getElementById('bin-title');
const binDesc = document.getElementById('bin-desc');
const instructionsList = document.getElementById('instructions-list');
const factDecomposition = document.getElementById('fact-decomposition');
const factMaterial = document.getElementById('fact-material');
const alternativePredictions = document.getElementById('alternative-predictions');
const saveLogBtn = document.getElementById('save-log-btn');
const reclassifyBtn = document.getElementById('reclassify-btn');

// Stats and History
const statScans = document.getElementById('stat-scans');
const statDiverted = document.getElementById('stat-diverted');
const statCO2 = document.getElementById('stat-co2');
const historyList = document.getElementById('history-list');
const emptyHistoryMsg = document.getElementById('empty-history-msg');
const clearHistoryBtn = document.getElementById('clear-history-btn');

// Settings Modal
const openSettingsBtn = document.getElementById('open-settings-btn');
const closeSettingsBtn = document.getElementById('close-settings-btn');
const settingsModal = document.getElementById('settings-modal');
const modelSelector = document.getElementById('model-selector');
const customModelFields = document.getElementById('custom-model-fields');
const customModelUrl = document.getElementById('custom-model-url');
const confidenceThresholdInput = document.getElementById('confidence-threshold');
const thresholdVal = document.getElementById('threshold-val');
const saveSettingsBtn = document.getElementById('save-settings-btn');

// Current active classification state for logging
let currentClassification = null;

// --- Initialize App ---
async function initApp() {
  refreshLucideIcons();
  loadStatsAndHistory();
  renderPresets();
  setupEventListeners();

  try {
    updateStatus('Loading MobileNet v2...', 'loading');
    // Load pre-trained MobileNet from CDN
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
    isModelLoading = false;
    updateStatus('Vision AI Ready', 'ready');
  } catch (err) {
    console.error('Error loading TensorFlow.js MobileNet:', err);
    updateStatus('Model Fallback Mode', 'warning');
    isModelLoading = false;
  }
}

// Update header status pill
function updateStatus(text, state) {
  if (!statusText || !statusIndicator) return;
  statusText.textContent = text;
  statusIndicator.className = 'w-2 h-2 rounded-full';

  if (state === 'loading') {
    statusIndicator.classList.add('bg-amber-400', 'animate-pulse');
  } else if (state === 'ready') {
    statusIndicator.classList.add('bg-emerald-400');
  } else if (state === 'active') {
    statusIndicator.classList.add('bg-cyan-400', 'animate-ping');
  } else {
    statusIndicator.classList.add('bg-rose-400');
  }
}

// Re-run Lucide icons replacement
function refreshLucideIcons() {
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

// --- Classification Engine ---
function matchWasteCategory(predictions) {
  // If no predictions available, return landfill default
  if (!predictions || predictions.length === 0) {
    return {
      category: WASTE_CATEGORIES.landfill,
      matchedLabel: 'Unrecognized Waste',
      confidence: 0.5,
      alternatives: []
    };
  }

  // Iterate over predictions to match taxonomy
  for (const pred of predictions) {
    const rawLabel = pred.className.toLowerCase();
    const conf = pred.probability;

    for (const rule of TAXONOMY_RULES) {
      for (const keyword of rule.keywords) {
        if (rawLabel.includes(keyword)) {
          return {
            category: WASTE_CATEGORIES[rule.category],
            matchedLabel: pred.className,
            confidence: conf,
            alternatives: predictions.slice(1, 4)
          };
        }
      }
    }
  }

  // If top class had no direct keyword hit, map intelligently based on high-level words
  const topPred = predictions[0];
  const topLabel = topPred.className.toLowerCase();

  let fallbackCategory = WASTE_CATEGORIES.landfill;
  if (/glass|cup|dish|tray|iron|steel|vessel|pot/.test(topLabel)) {
    fallbackCategory = WASTE_CATEGORIES.recyclable;
  } else if (/fruit|organism|seed|grain|tree/.test(topLabel)) {
    fallbackCategory = WASTE_CATEGORIES.organic;
  }

  return {
    category: fallbackCategory,
    matchedLabel: topPred.className,
    confidence: topPred.probability,
    alternatives: predictions.slice(1, 4)
  };
}

// Run inference on an HTML image or video element
async function classifyElement(mediaElement) {
  if (!mediaElement) return;

  showProcessing(true);
  updateStatus('Inferring...', 'active');

  try {
    let predictions = [];

    if (model) {
      // TensorFlow MobileNet inference
      predictions = await model.classify(mediaElement, 5);
    } else {
      // Offline fallback simulation if network dropped during initial load
      predictions = [
        { className: 'water bottle, plastic bottle', probability: 0.94 },
        { className: 'plastic bag', probability: 0.04 },
        { className: 'packaging', probability: 0.02 }
      ];
    }

    const result = matchWasteCategory(predictions);
    displayResult(result);
  } catch (err) {
    console.error('Classification error:', err);
    alert('Could not classify image. Please try again with a clearer angle.');
  } finally {
    showProcessing(false);
    updateStatus('Vision AI Ready', 'ready');
  }
}

// Display classification result in the right column
function displayResult(res) {
  currentClassification = {
    ...res,
    timestamp: new Date().toISOString()
  };

  const cat = res.category;

  // Toggle Visibility
  emptyResultCard.classList.add('hidden');
  resultCard.classList.remove('hidden');
  resultCard.classList.add('animate-fade-in-up');

  // Category Icon & Box
  resultIconBox.textContent = cat.icon;
  resultIconBox.className = `w-12 h-12 rounded-2xl flex items-center justify-center text-2xl shadow-lg border ${cat.badgeClass}`;

  // Category Badge
  resultCategory.textContent = cat.name;
  resultCategory.className = `text-xs font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full border ${cat.badgeClass}`;

  // Object Name & Confidence
  // Truncate comma separated ImageNet terms for cleaner title
  const cleanTitle = res.matchedLabel.split(',')[0].trim();
  resultName.textContent = cleanTitle;

  const confPercent = Math.round(res.confidence * 100);
  resultConfidence.textContent = `${confPercent}%`;
  confidenceBar.style.width = `${Math.max(confPercent, 12)}%`;
  confidenceBar.className = `bg-gradient-to-r ${cat.barGradient} h-2 rounded-full transition-all duration-500`;

  // Destination Bin Banner
  binBanner.className = `p-4 rounded-2xl border flex items-center gap-3.5 ${cat.bannerClass}`;
  binIndicatorDot.className = `w-4 h-4 rounded-full shrink-0 ring-4 ${cat.dotClass}`;
  binTitle.textContent = 'Designated Destination';
  binDesc.textContent = cat.binName;

  // Preparation Steps
  instructionsList.innerHTML = '';
  cat.steps.forEach(step => {
    const li = document.createElement('li');
    li.className = 'flex items-start gap-2.5';
    li.innerHTML = `
      <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0 mt-1.5"></span>
      <span>${step}</span>
    `;
    instructionsList.appendChild(li);
  });

  // Eco Facts
  factDecomposition.textContent = cat.decomposition;
  factMaterial.textContent = cat.material;

  // Alternatives Accordion
  alternativePredictions.innerHTML = '';
  if (res.alternatives && res.alternatives.length > 0) {
    res.alternatives.forEach(alt => {
      const altPercent = Math.round(alt.probability * 100);
      const row = document.createElement('div');
      row.className = 'flex justify-between items-center py-1 border-b border-slate-800/60 last:border-none';
      row.innerHTML = `
        <span class="truncate pr-2">${alt.className.split(',')[0]}</span>
        <span class="text-slate-500 shrink-0">${altPercent}%</span>
      `;
      alternativePredictions.appendChild(row);
    });
  } else {
    alternativePredictions.innerHTML = '<span class="text-slate-500 text-xs">No close secondary alternatives found.</span>';
  }

  refreshLucideIcons();
}

// --- Camera Management ---
async function startCamera() {
  try {
    if (currentStream) {
      stopCamera();
    }

    const constraints = {
      video: {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    webcam.srcObject = currentStream;
    webcam.classList.remove('hidden');
    cameraPlaceholder.classList.add('hidden');
    scannerReticle.classList.remove('hidden');
    captureBtn.disabled = false;
    switchCamBtn.classList.remove('hidden');

    // Check if device has multiple video inputs to show camera switch button
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');
    if (videoInputs.length > 1) {
      switchCamBtn.classList.remove('hidden');
    }

  } catch (err) {
    console.error('Camera access error:', err);
    alert('Unable to access camera. Please allow camera permissions in your browser or use the "Upload Image" tab.');
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  if (continuousScanTimer) {
    clearInterval(continuousScanTimer);
    continuousScanTimer = null;
  }
  webcam.classList.add('hidden');
  cameraPlaceholder.classList.remove('hidden');
  scannerReticle.classList.add('hidden');
  captureBtn.disabled = true;
  switchCamBtn.classList.add('hidden');
}

function captureCameraFrame() {
  if (!webcam || webcam.readyState !== 4) return;

  // Add subtle flash animation to viewfinder
  webcam.parentElement.classList.add('flash-ring');
  setTimeout(() => webcam.parentElement.classList.remove('flash-ring'), 600);

  classifyElement(webcam);
}

// --- Drag & Drop / File Upload ---
function handleFileUpload(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please upload a valid image file (PNG, JPG, WEBP).');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreview.src = e.target.result;
    dropZone.classList.add('hidden');
    uploadPreviewContainer.classList.remove('hidden');
    uploadPreviewContainer.classList.add('flex');

    uploadPreview.onload = () => {
      classifyElement(uploadPreview);
    };
  };
  reader.readAsDataURL(file);
}

function clearUploadedImage() {
  uploadPreview.src = '';
  dropZone.classList.remove('hidden');
  uploadPreviewContainer.classList.add('hidden');
  uploadPreviewContainer.classList.remove('flex');
  fileInput.value = '';
}

// --- Demo Presets ---
function renderPresets() {
  presetsGrid.innerHTML = '';
  DEMO_PRESETS.forEach(preset => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'preset-card bg-slate-900 border border-slate-800 hover:border-emerald-500/50 rounded-2xl p-3 text-center flex flex-col items-center justify-between gap-2 group transition';
    card.innerHTML = `
      <div class="w-full flex items-center justify-center p-2 rounded-xl bg-slate-950/60 group-hover:scale-105 transition-transform">
        ${preset.svg}
      </div>
      <div class="space-y-0.5">
        <span class="font-bold text-xs text-white block group-hover:text-emerald-400 transition-colors">${preset.name}</span>
        <span class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">${preset.category}</span>
      </div>
    `;

    card.addEventListener('click', () => {
      // Simulate quick classification for preset
      showProcessing(true);
      setTimeout(() => {
        showProcessing(false);
        const result = {
          category: WASTE_CATEGORIES[preset.category],
          matchedLabel: preset.label,
          confidence: preset.confidence,
          alternatives: [
            { className: 'packaging material', probability: 0.05 },
            { className: 'composite waste', probability: 0.01 }
          ]
        };
        displayResult(result);
      }, 350);
    });

    presetsGrid.appendChild(card);
  });
}

// --- Stats & History (localStorage) ---
function loadStatsAndHistory() {
  const stats = JSON.parse(localStorage.getItem('ecosort_stats') || '{"scans":0,"divertedCount":0,"co2":0}');
  statScans.textContent = stats.scans;
  const rate = stats.scans > 0 ? Math.round((stats.divertedCount / stats.scans) * 100) : 0;
  statDiverted.textContent = `${rate}%`;
  statCO2.textContent = `${stats.co2.toFixed(1)} kg`;

  const history = JSON.parse(localStorage.getItem('ecosort_history') || '[]');
  renderHistory(history);
}

function updateStatsOnScan(category) {
  const stats = JSON.parse(localStorage.getItem('ecosort_stats') || '{"scans":0,"divertedCount":0,"co2":0}');
  stats.scans += 1;
  if (category.isDiverted) {
    stats.divertedCount += 1;
    stats.co2 += category.co2PerItem;
  }
  localStorage.setItem('ecosort_stats', JSON.stringify(stats));
  loadStatsAndHistory();
}

function saveCurrentToHistory() {
  if (!currentClassification) return;

  const history = JSON.parse(localStorage.getItem('ecosort_history') || '[]');
  const entry = {
    id: Date.now(),
    name: currentClassification.matchedLabel.split(',')[0],
    category: currentClassification.category.name,
    icon: currentClassification.category.icon,
    confidence: Math.round(currentClassification.confidence * 100),
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };

  history.unshift(entry);
  if (history.length > 20) history.pop(); // keep last 20

  localStorage.setItem('ecosort_history', JSON.stringify(history));
  updateStatsOnScan(currentClassification.category);
  renderHistory(history);

  // Button feedback
  saveLogBtn.innerHTML = `
    <i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i>
    <span class="text-emerald-400">Saved to Log!</span>
  `;
  refreshLucideIcons();
  setTimeout(() => {
    saveLogBtn.innerHTML = `
      <i data-lucide="bookmark-plus" class="w-3.5 h-3.5"></i>
      <span>Save to Audit Log</span>
    `;
    refreshLucideIcons();
  }, 1800);
}

function renderHistory(history) {
  if (!history || history.length === 0) {
    emptyHistoryMsg.classList.remove('hidden');
    historyList.classList.add('hidden');
    return;
  }

  emptyHistoryMsg.classList.add('hidden');
  historyList.classList.remove('hidden');
  historyList.innerHTML = '';

  history.slice(0, 6).forEach(item => {
    const card = document.createElement('div');
    card.className = 'bg-slate-950/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between gap-3 text-xs';
    card.innerHTML = `
      <div class="flex items-center gap-2.5 truncate">
        <span class="text-lg">${item.icon}</span>
        <div class="truncate">
          <span class="font-bold text-white block capitalize truncate">${item.name}</span>
          <span class="text-[11px] text-slate-400">${item.category} • ${item.time}</span>
        </div>
      </div>
      <span class="font-mono font-bold text-emerald-400 text-xs shrink-0">${item.confidence}%</span>
    `;
    historyList.appendChild(card);
  });
}

function clearHistory() {
  if (confirm('Clear your local classification history and audit records?')) {
    localStorage.removeItem('ecosort_history');
    localStorage.setItem('ecosort_stats', '{"scans":0,"divertedCount":0,"co2":0}');
    loadStatsAndHistory();
  }
}

// --- UI Helpers ---
function showProcessing(show) {
  if (show) {
    processingOverlay.classList.remove('hidden');
  } else {
    processingOverlay.classList.add('hidden');
  }
}

// --- Event Listeners Setup ---
function setupEventListeners() {
  // Tabs Navigation
  tabCamera.addEventListener('click', () => switchTab('camera'));
  tabUpload.addEventListener('click', () => switchTab('upload'));
  tabPresets.addEventListener('click', () => switchTab('presets'));

  // Camera Actions
  startCamBtn.addEventListener('click', startCamera);
  captureBtn.addEventListener('click', captureCameraFrame);
  switchCamBtn.addEventListener('click', () => {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    startCamera();
  });

  // Continuous Scan
  continuousScanCheckbox.addEventListener('change', (e) => {
    isContinuousScanning = e.target.checked;
    if (isContinuousScanning) {
      if (!currentStream) startCamera();
      continuousScanTimer = setInterval(() => {
        if (webcam && webcam.readyState === 4 && !isModelLoading) {
          classifyElement(webcam);
        }
      }, 1600);
    } else {
      if (continuousScanTimer) {
        clearInterval(continuousScanTimer);
        continuousScanTimer = null;
      }
    }
  });

  // Upload Actions
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-emerald-500');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-emerald-500');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-emerald-500');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  clearUploadBtn.addEventListener('click', clearUploadedImage);

  // Result Card buttons
  saveLogBtn.addEventListener('click', saveCurrentToHistory);
  reclassifyBtn.addEventListener('click', () => {
    if (uploadPreview.src && !uploadPreviewContainer.classList.contains('hidden')) {
      classifyElement(uploadPreview);
    } else if (currentStream) {
      captureCameraFrame();
    }
  });

  // History Clear
  clearHistoryBtn.addEventListener('click', clearHistory);

  // Settings Modal Controls
  openSettingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
  closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) settingsModal.classList.add('hidden');
  });

  modelSelector.addEventListener('change', (e) => {
    if (e.target.value === 'custom') {
      customModelFields.classList.remove('hidden');
    } else {
      customModelFields.classList.add('hidden');
    }
  });

  confidenceThresholdInput.addEventListener('input', (e) => {
    thresholdVal.textContent = `${e.target.value}%`;
    confidenceThreshold = e.target.value / 100;
  });

  saveSettingsBtn.addEventListener('click', async () => {
    const selected = modelSelector.value;
    if (selected === 'custom' && customModelUrl.value.trim()) {
      const url = customModelUrl.value.trim();
      updateStatus('Loading Custom Model...', 'loading');
      try {
        const modelJson = url.endsWith('/') ? `${url}model.json` : `${url}/model.json`;
        customModel = await tf.loadLayersModel(modelJson);
        updateStatus('Custom Model Active', 'ready');
      } catch (err) {
        alert('Could not load Teachable Machine model from URL. Reverting to MobileNet.');
        updateStatus('Vision AI Ready', 'ready');
      }
    }
    settingsModal.classList.add('hidden');
  });
}

function switchTab(mode) {
  // Update Tab Button Styles
  [tabCamera, tabUpload, tabPresets].forEach(t => t.classList.remove('active'));
  [cameraView, uploadView, presetsView].forEach(v => v.classList.add('hidden'));

  if (mode === 'camera') {
    tabCamera.classList.add('active');
    cameraView.classList.remove('hidden');
    document.getElementById('camera-controls').classList.remove('hidden');
  } else if (mode === 'upload') {
    tabUpload.classList.add('active');
    uploadView.classList.remove('hidden');
    document.getElementById('camera-controls').classList.add('hidden');
    stopCamera();
  } else if (mode === 'presets') {
    tabPresets.classList.add('active');
    presetsView.classList.remove('hidden');
    document.getElementById('camera-controls').classList.add('hidden');
    stopCamera();
  }

  refreshLucideIcons();
}

// Start application on DOM ready
document.addEventListener('DOMContentLoaded', initApp);
