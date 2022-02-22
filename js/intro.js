import {fgsmTargeted, bimTargeted, jsmaOnePixel, jsma, cw} from './adversarial.js';
import {CIFAR_CLASSES} from './class_names.js';
import {vote_with_masks, vote_with_noises} from './vote.js';



class Subtract extends tf.layers.Layer {
  constructor() {
    super({});
  }
  // In this case, the output is a scalar.
  computeOutputShape(inputShape) { return inputShape[0]; }
 
  // call() is where we do the computation.
  call(input, kwargs) { return input[0].sub(input[1]);}
 
  // Every layer needs a unique name.
  static get className() { return 'Subtract'; }
 }

 tf.serialization.registerClass(Subtract);

/************************************************************************
* Global constants
************************************************************************/

const $ = query => document.querySelector(query);

const CIFAR_CONFIGS = {
  'fgsm': {ε: 0.05},  // 0.1 L_inf perturbation is too visible in color
  'jsmaOnePixel': {ε: 75},  // JSMA one-pixel on CIFAR-10 requires more ~3x pixels than MNIST
  'jsma': {ε: 75},  // JSMA on CIFAR-10 also requires more ~3x pixels than MNIST
  'cw': {c: 1, λ: 0.05}  // Tried to minimize distortion, but not sure it worked
};

/************************************************************************
* Load Datasets
************************************************************************/


/****************************** Load CIFAR-10 ******************************/

let cifarXUrl = 'data/cifar/cifar10_sample_x.json';
let cifarYUrl = 'data/cifar/cifar10_sample_y.json';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let cifarX, cifarY, cifarDataset;
let loadingCifarX = fetch(cifarXUrl).then(res => res.json()).then(arr => cifarX = tf.data.array(arr).batch(1));
let loadingCifarY = fetch(cifarYUrl).then(res => res.json()).then(arr => cifarY = tf.data.array(arr).batch(1));
let loadingCifar = Promise.all([loadingCifarX, loadingCifarY]).then(() => tf.data.zip([cifarX, cifarY]).toArray()).then(ds => cifarDataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));



/************************************************************************
* Load Models
************************************************************************/


/****************************** Load CIFAR-10 ******************************/

let cifarModel;
async function loadCifarModel() {
  if (cifarModel !== undefined) { return; }
  cifarModel = await tf.loadLayersModel('data/cifar/cifar10_cnn_new.json');
}




/************************************************************************
* Load Denoised Models
************************************************************************/

/****************************** Load CIFAR-10 ******************************/

let cifarDenoisedModel;
async function loadCifarDenoisedModel() {
  if (cifarDenoisedModel !== undefined) { return; }
  cifarDenoisedModel = await tf.loadLayersModel('data/denoiser/cifar10_denoiser_mask.json');
}


/************************************************************************
* Attach Event Handlers
************************************************************************/

// On page load
window.addEventListener('load', showImage);
window.addEventListener('load', resetAvailableAttacks);
window.addEventListener('load', showBanners);
window.addEventListener('load', removeLeftOverlay);

// Model selection dropdown
// $('#select-model').addEventListener('change', showImage);
// $('#select-model').addEventListener('change', resetOnNewImage);
// $('#select-model').addEventListener('change', resetAttack);
// $('#select-model').addEventListener('change', removeLeftOverlay);

// Next image button
$('#next-image').addEventListener('click', showNextImage);
$('#next-image').addEventListener('click', resetOnNewImage);
$('#next-image').addEventListener('click', resetAttack);

// Predict button (original image)
$('#predict-original').addEventListener('click', predict);
$('#predict-original').addEventListener('click', removeTopMidOverlay);

// Target label dropdown
$('#select-target').addEventListener('change', resetAttack);

// Attack algorithm dropdown
$('#select-attack').addEventListener('change', resetAttack);

// Generate button
$('#generate-adv').addEventListener('click', generateAdv);
$('#generate-adv').addEventListener('click', removeBottomMidOverlay);

// Predict button (adversarial image)
$('#predict-adv').addEventListener('click', predictAdv);
$('#predict-adv').addEventListener('click', removeTopROverlay);

// Generate button (denoised image)
$('#generate-denoised').addEventListener('click', generateDenoised);
$('#generate-denoised').addEventListener('click', removeBottomROverlay);

// Predict button (denoised image)
$('#predict-denoised').addEventListener('click', predictDenoised);


// View noise / view image link
$('#view-noise').addEventListener('click', viewNoise);
$('#view-image').addEventListener('click', viewImage);

// View denoised / view masked link
$('#view-masked').addEventListener('click', viewMasked);
$('#view-denoised').addEventListener('click', viewDenoised);


/************************************************************************
* Define Event Handlers
************************************************************************/

/**
 * Renders the next image from the sample dataset in the original canvas
 */
function showNextImage() {
  showNextCifar();
}

/**
 * Renders the current image from the sample dataset in the original canvas
 */
function showImage() {
  showCifar();
}

/**
 * Computes & displays prediction of the current original image
 */
async function predict() {
  $('#predict-original').disabled = true;
  $('#predict-original').innerText = 'Loading...';

  await loadCifarModel();
  await loadingCifar;
  let lblIdx = cifarDataset[cifarIdx].ys.argMax(1).dataSync()[0];
  _predict(cifarModel, cifarDataset[cifarIdx].xs, lblIdx, CIFAR_CLASSES);

  $('#predict-original').innerText = 'Run Neural Network';

  function _predict(model, img, lblIdx, CLASS_NAMES) {
    // Generate prediction
    // let pred = model.predict(img);
    let pred = model.predict(img);

    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    let status = {msg: '✅ Prediction is correct.', statusClass: 'status-green'};  // Predictions on the sample should always be correct
    showPrediction(`Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${(predProb * 100).toFixed(2)}%`, status);
  }
 }

/**
 * Generates adversarial example from the current original image
 */
let advPrediction, advStatus;
let adv_img;
async function generateAdv() {
  $('#generate-adv').disabled = true;
  $('#generate-adv').innerText = 'Loading...';

  let attack;
  switch ($('#select-attack').value) {
    case 'fgsmTargeted': attack = fgsmTargeted; break;
    case 'bimTargeted': attack = bimTargeted; break;
    case 'jsmaOnePixel': attack = jsmaOnePixel; break;
    case 'jsma': attack = jsma; break;
    case 'cw': attack = cw; break;
  }

  let targetLblIdx = parseInt($('#select-target').value);
  await loadCifarModel();
  await loadingCifar;
  await _generateAdv(cifarModel, cifarDataset[cifarIdx].xs, cifarDataset[cifarIdx].ys, CIFAR_CLASSES, CIFAR_CONFIGS[attack.name]);

  $('#latency-msg').style.display = 'none';
  $('#generate-adv').innerText = 'Generate';
  $('#predict-adv').innerText = 'Run Neural Network';
  $('#predict-adv').disabled = false;

  async function _generateAdv(model, img, lbl, CLASS_NAMES, CONFIG) {
    // Generate adversarial example
    let targetLbl = tf.oneHot(targetLblIdx, lbl.shape[1]).reshape(lbl.shape);
    let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl, CONFIG));
    adv_img = aimg
    // Display adversarial example
    $('#difference').style.display = 'block';
    await drawImg(aimg, 'adversarial');

    // Compute & store adversarial prediction
    let pred = model.predict(aimg);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];
    advPrediction = `Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${(predProb * 100).toFixed(2)}%`;

    // Compute & store attack success/failure message
    let lblIdx = lbl.argMax(1).dataSync()[0];
    if (predLblIdx === targetLblIdx) {
      advStatus = {msg: '❌ Prediction is wrong. Attack succeeded!', statusClass: 'status-red'};
    } else if (predLblIdx !== lblIdx) {
      advStatus = {msg: '❌ Prediction is wrong. Attack partially succeeded!', statusClass: 'status-orange'};
    } else {
      advStatus = {msg: '✅ Prediction is still correct. Attack failed.', statusClass: 'status-green'};
    }

    // Also compute and draw the adversarial noise (hidden until the user clicks on it)
    let noise = tf.sub(aimg, img).add(0.5).clipByValue(0, 1);  // [Szegedy 14] Intriguing properties of neural networks
    drawImg(noise, 'adversarial-noise');
  }
}

let denoisedPrediction, denoisedStatus;
async function generateDenoised() {
  $('#generate-denoised').disabled = true;
  $('#generate-denoised').innerText = 'Loading...';

  await loadCifarModel();
  await loadingCifar;
  await loadCifarDenoisedModel();
  await _generateDenoised(cifarModel, cifarDenoisedModel, adv_img, cifarDataset[cifarIdx].ys, CIFAR_CLASSES);
  $('#latency-denoised-msg').style.display = 'none';
  $('#generate-denoised').innerText = 'Generate';
  $('#predict-denoised').innerText = 'Run Neural Network';
  $('#predict-denoised').disabled = false;

  async function _generateDenoised(model, denoised_model, img, lbl, CLASS_NAMES) {
    // Generate denoised example
    // let dimg = tf.tidy(() => denoised_model.predict(img));
    
    // let vote_results = tf.tidy(() => vote_with_noises(model, denoised_model, img, 100));
    let vote_results = tf.tidy(() => vote_with_masks(model, denoised_model, img, 100));

    let predLblIdx = vote_results[0];
    let predProb = vote_results[1]/100;

    let mimg = vote_results[2].slice([0], [1]);
    let dimg = vote_results[3].slice([0], [1]);
    // Display denoised example
    $('#difference-denoised').style.display = 'block';
    await drawImg(dimg, 'denoised');
  
    // Compute & store denoised prediction
    // let pred = model.predict(dimg);
    // let predLblIdx = pred.argMax(1).dataSync()[0];
    // let predProb = pred.max().dataSync()[0];

    denoisedPrediction = `Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${(predProb * 100).toFixed(2)}%`;

    // Compute & store attack success/failure message
    let lblIdx = lbl.argMax(1).dataSync()[0];
    if (predLblIdx === lblIdx) {
      denoisedStatus = {msg: '✅ Prediction is correct. Purify succeeded.', statusClass: 'status-green'};
    } else {
      denoisedStatus = {msg: '❌ Prediction is still wrong. Purify failed!', statusClass: 'status-red'};
    }
    
    drawImg(mimg, 'masked');
  }
}


/**
 * Displays prediction for the current adversarial image
 * (This function just renders the status we've already computed in generateAdv())
 */
function predictAdv() {
  $('#predict-adv').disabled = true;
  showAdvPrediction(advPrediction, advStatus);
}


/**
 * Displays prediction for the current denoised image
 * (This function just renders the status we've already computed in generateDenoised())
 */
 function predictDenoised() {
  $('#predict-denoised').disabled = true;
  showDenoisedPrediction(denoisedPrediction, denoisedStatus);
}

/**
 * Show adversarial noise when the user clicks on the "view noise" link
 */
async function viewNoise() {
  $('#difference').style.display = 'none';
  $('#difference-noise').style.display = 'block';
  $('#adversarial').style.display = 'none';
  $('#adversarial-noise').style.display = 'block';
}

/**
 * Show adversarial image when the user clicks on the "view image" link
 */
async function viewImage() {
  $('#difference').style.display = 'block';
  $('#difference-noise').style.display = 'none';
  $('#adversarial').style.display = 'block';
  $('#adversarial-noise').style.display = 'none';
}


/**
 * Show adversarial noise when the user clicks on the "view noise" link
 */
 async function viewDenoised() {
  $('#difference-masked').style.display = 'none';
  $('#difference-denoised').style.display = 'block';
  $('#masked').style.display = 'none';
  $('#denoised').style.display = 'inline-block';
}

/**
 * Show adversarial image when the user clicks on the "view image" link
 */
async function viewMasked() {
  $('#difference-masked').style.display = 'block';
  $('#difference-denoised').style.display = 'none';
  $('#masked').style.display = 'inline-block';
  $('#denoised').style.display = 'none';
}



/**
 * Reset entire dashboard UI when a new image is selected
 */
function resetOnNewImage() {
  $('#predict-original').disabled = false;
  $('#predict-original').innerText = 'Run Neural Network';
  $('#prediction').style.display = 'none';
  $('#prediction-status').innerHTML = '';
  $('#prediction-status').className = '';
  $('#prediction-status').style.marginBottom = '9px';
  resetAttack();
  resetAvailableAttacks();
}

/**
 * Reset attack UI when a new target label, attack, or image is selected
 */
async function resetAttack() {
  $('#generate-adv').disabled = false;
  $('#predict-adv').disabled = true;
  $('#predict-adv').innerText = 'Click "Generate" First';
  $('#difference').style.display = 'none';
  $('#difference-noise').style.display = 'none';
  $('#prediction-adv').style.display = 'none';
  $('#prediction-adv-status').innerHTML = '';
  $('#prediction-adv-status').className = '';
  $('#prediction-adv-status').style.marginBottom = '9px';
  await drawImg(tf.ones([1, 224, 224, 1]), 'adversarial');
  await drawImg(tf.ones([1, 224, 224, 1]), 'adversarial-noise');
  $('#adversarial').style.display = 'block';
  $('#adversarial-noise').style.display = 'none';
  // $('#adversarial-image-overlay').style.display = 'block';
  // $('#adversarial-canvas-overlay').style.display = 'block';
  // $('#adversarial-prediction-overlay').style.display = 'block';
  $('#latency-msg').style.display = 'none';
  resetDenoised();
}


/**
 * Reset denoised UI when a new target label, attack, or image is selected
 */
 async function resetDenoised() {
  $('#generate-denoised').disabled = false;
  $('#predict-denoised').disabled = true;
  $('#predict-denoised').innerText = 'Click "Generate" First';
  $('#difference-denoised').style.display = 'none';
  $('#difference-masked').style.display = 'none';
  $('#prediction-denoised').style.display = 'none';
  $('#prediction-denoised-status').innerHTML = '';
  $('#prediction-denoised-status').className = '';
  $('#prediction-denoised-status').style.marginBottom = '9px';
  // $('#denoised-image-overlay').style.display = 'block';
  // $('#denoised-canvas-overlay').style.display = 'block';
  // $('#denoised-prediction-overlay').style.display = 'block';

  await drawImg(tf.ones([1, 224, 224, 1]), 'denoised');
  await drawImg(tf.ones([1, 224, 224, 1]), 'masked');

  $('#denoised').style.display = 'inline-block';
  $('#masked').style.display = 'none';

}

/**
 * Reset available attacks and target labels when a new image is selected
 */
function resetAvailableAttacks() {
  const CIFAR_TARGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  let originalLbl = cifarDataset[cifarIdx].ys.argMax(1).dataSync()[0];
  _resetAvailableAttacks(true, originalLbl, CIFAR_TARGETS, CIFAR_CLASSES);


  function _resetAvailableAttacks(jsma, originalLbl, TARGETS, CLASS_NAMES) {
    let modelName = $('#select-model').value;
    let selectAttack = $('#select-attack');
    let selectTarget = $('#select-target');

    // Add or remove JSMA as an option
    if (jsma === true) {
      selectAttack.querySelector('option[value=jsma]').disabled = false;
    } else {
      selectAttack.querySelector('option[value=jsma]').disabled = true;
      if (selectAttack.value === 'jsma') { selectAttack.value = 'fgsmTargeted'; }
    }

    // Filter available target classes in dropdown
    if (selectTarget.getAttribute('data-model') === modelName) {
      // Go through options and disable the current class as a target class
      selectTarget.options.forEach(option => {
        if (parseInt(option.value) === originalLbl) { option.disabled = true; }
        else {option.disabled = false; }
      });
      // Reset the selected option if it's now disabled
      if (parseInt(selectTarget.value) === originalLbl) {
        selectTarget.options[0].selected = true;
        if (parseInt(selectTarget.value) === originalLbl) {
          selectTarget.options[1].selected = true;
        }
      }
    } else {
      // Rebuild options from scratch (b/c the user chose a new model)
      selectTarget.innerHTML = '';
      TARGETS.forEach(i => {
        let option = new Option(CLASS_NAMES[i], i);
        if (i === originalLbl) { option.disabled = true; }
        selectTarget.appendChild(option);
      });
      selectTarget.setAttribute('data-model', modelName);
    }
  }
}

/**
 * Removes the overlay on the left half of the dashboard when the user selects a model
 */
function removeLeftOverlay() {
  $('#original-image-overlay').style.display = 'none';
  $('#original-canvas-overlay').style.display = 'none';
  $('#original-prediction-overlay').style.display = 'none';
}

/**
 * Removes the overlay on the top right of the dashboard when the user makes their first prediction
 */
function removeTopMidOverlay() {
  $('#adversarial-image-overlay').style.display = 'none';
  $('#adversarial-canvas-overlay').style.display = 'none';
}

/**
 * Removes the overlay on the bottom right of the dashboard when the user generates an adversarial example
 */
function removeBottomMidOverlay() {
  $('#adversarial-prediction-overlay').style.display = 'none';
}

/**
 * Removes the overlay on the top right of the dashboard when the user makes their first prediction
 */
 function removeTopROverlay() {
  $('#denoised-image-overlay').style.display = 'none';
  $('#denoised-canvas-overlay').style.display = 'none';
}

/**
 * Removes the overlay on the bottom right of the dashboard when the user generates an adversarial example
 */
function removeBottomROverlay() {
  $('#denoised-prediction-overlay').style.display = 'none';
}

/**
 * Check the user's device and display appropriate warning messages
 */
function showBanners() {
  if (!supports32BitWebGL()) { $('#mobile-banner').style.display = 'block'; }
  else if (!isDesktopChrome()) { $('#chrome-banner').style.display = 'block'; }
}

/**
 * Returns if it looks like the user is on desktop Google Chrome
 * https://stackoverflow.com/a/13348618/908744
 */
function isDesktopChrome() {
  var isChromium = window.chrome;
  var winNav = window.navigator;
  var vendorName = winNav.vendor;
  var isOpera = typeof window.opr !== "undefined";
  var isIEedge = winNav.userAgent.indexOf("Edge") > -1;
  var isIOSChrome = winNav.userAgent.match("CriOS");

  if (isIOSChrome) {
    return false;
  } else if (
    isChromium !== null &&
    typeof isChromium !== "undefined" &&
    vendorName === "Google Inc." &&
    isOpera === false &&
    isIEedge === false
  ) {
    return true;
  } else {
    return false;
  }
}

/**
 * Returns if the current device supports WebGL 32-bit
 * https://www.tensorflow.org/js/guide/platform_environment#precision
 */
function supports32BitWebGL() {
  return tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE') && tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');
}

/************************************************************************
* Visualize Images
************************************************************************/

function showPrediction(msg, status) {
  $('#prediction').innerHTML = msg;
  $('#prediction').style.display = 'block';
  $('#prediction-status').innerHTML = status.msg;
  $('#prediction-status').className = status.statusClass;
  $('#prediction-status').style.marginBottom = '15px';
}

function showAdvPrediction(msg, status) {
  $('#prediction-adv').innerHTML = msg;
  $('#prediction-adv').style.display = 'block';
  $('#prediction-adv-status').innerHTML = status.msg;
  $('#prediction-adv-status').className = status.statusClass;
  $('#prediction-adv-status').style.marginBottom = '15px';
}

function showDenoisedPrediction(msg, status) {
  $('#prediction-denoised').innerHTML = msg;
  $('#prediction-denoised').style.display = 'block';
  $('#prediction-denoised-status').innerHTML = status.msg;
  $('#prediction-denoised-status').className = status.statusClass;
  $('#prediction-denoised-status').style.marginBottom = '15px';
}


let cifarIdx = 0;
async function showCifar() {
  await loadingCifar;
  await drawImg(cifarDataset[cifarIdx].xs, 'original');
}
async function showNextCifar() {
  cifarIdx = (cifarIdx + 1) % cifarDataset.length;
  await showCifar();
}


async function drawImg(img, element) {
  // Draw image
  let canvas = document.getElementById(element);
  if (img.shape[0] === 1) { img = img.squeeze(0); }
  if (img.shape[0] === 784) {
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([28, 28, 1]), [224, 224]);
    await tf.browser.toPixels(resizedImg, canvas);
  } else {
    let resizedImg = tf.image.resizeNearestNeighbor(img, [224, 224]);
    await tf.browser.toPixels(resizedImg, canvas);
  }
}
