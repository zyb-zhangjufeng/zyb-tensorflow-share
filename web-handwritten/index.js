/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as tf from "@tensorflow/tfjs";

const HANDWRITTEN_MODEL_PATH = "./web_model/model.json";

const IMAGE_SIZE = 180;
const TOPK_PREDICTIONS = 2;

const IMAGENET_CLASSES = ["手写", "非手写"];

const localImgs = ["cat", "handwritten"];

let handwritten;
const handwrittenDemo = async () => {
  status("加载模型...");

  handwritten = await tf.loadGraphModel(HANDWRITTEN_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  handwritten.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status("");

  // Make a prediction through the locally hosted
  for (const localImg of localImgs) {
    const imgElement = document.getElementById(localImg);
    if (imgElement.complete && imgElement.naturalHeight !== 0) {
      predict(imgElement);
      imgElement.style.display = "";
    } else {
      imgElement.onload = () => {
        predict(imgElement);
        imgElement.style.display = "";
      };
    }
  }

  document.getElementById("file-container").style.display = "";
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status("预测...");

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.cast(tf.browser.fromPixels(imgElement), "float32");

    // const offset = tf.scalar(127.5);
    // // Normalize the image from [0, 255] to [-1, 1].
    // const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return handwritten.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`${Math.floor(totalTime1)}ms 完成`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
  const softmaxLogits = tf.softmax(logits);
  const values = await softmaxLogits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i],
    });
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement("div");
  predictionContainer.className = "pred-container";

  if (classes[0].className === IMAGENET_CLASSES[0]) {
    predictionContainer.style.backgroundColor = "lightgreen";
  } else {
    predictionContainer.style.backgroundColor = "lightcoral";
  }

  const imgContainer = document.createElement("div");
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement("div");
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement("div");
    row.className = "row";

    const classElement = document.createElement("div");
    classElement.className = "cell";
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement("div");
    probsElement.className = "cell";
    probsElement.innerText = `${(classes[i].probability * 100).toFixed(1)}%`;
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
    predictionContainer,
    predictionsElement.firstChild
  );
}

const filesElement = document.getElementById("files");
filesElement.addEventListener("change", (evt) => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; (f = files[i]); i++) {
    // Only process image files (skip non image files)
    if (!f.type.match("image.*")) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = (e) => {
      // Fill the image & call predict.
      let img = document.createElement("img");
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById("status");
const status = (msg) => (demoStatusElement.innerText = msg);

const predictionsElement = document.getElementById("predictions");

handwrittenDemo();
