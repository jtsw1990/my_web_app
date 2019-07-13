var modelName = "digitrecognizercnn";
let model;


var canvasWidth = 450;
var canvasHeight = 450;
var canvasLineWidth = 40;

//var canvasWidth           	= 150;
//var canvasHeight 			= 150;
var canvasStrokeStyle		= "white";
var canvasLineJoin			= "round";
//var canvasLineWidth       	= 12;
var canvasBackgroundColor 	= "black";
var canvasId              	= "canvas";

var clickX = new Array();
var clickY = new Array();
var clickD = new Array();
var drawing;

document.getElementById('chart_box').innerHTML = "";
document.getElementById('chart_box').style.display = "none";

//---------------
// Create canvas
//---------------
var canvasBox = document.getElementById('canvas_box');
var canvas    = document.createElement("canvas");

canvas.setAttribute("width", canvasWidth);
canvas.setAttribute("height", canvasHeight);
canvas.setAttribute("id", canvasId);
canvas.style.backgroundColor = canvasBackgroundColor;
canvasBox.appendChild(canvas);
if(typeof G_vmlCanvasManager != 'undefined') {
  canvas = G_vmlCanvasManager.initElement(canvas);
}

ctx = canvas.getContext("2d");


modelName = "digitrecognizercnn";
loadModel(modelName);

//---------------------
// MOUSE DOWN function
//---------------------
$("#canvas").mousedown(function(e) {
	var mouseX = e.pageX - this.offsetLeft;
	var mouseY = e.pageY - this.offsetTop;

	drawing = true;
	addUserGesture(mouseX, mouseY);
	drawOnCanvas();
});

//-----------------------
// TOUCH START function
//-----------------------
canvas.addEventListener("touchstart", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}

	var rect = canvas.getBoundingClientRect();
	var touch = e.touches[0];

	var mouseX = touch.clientX - rect.left;
	var mouseY = touch.clientY - rect.top;

	drawing = true;
	addUserGesture(mouseX, mouseY);
	drawOnCanvas();

}, false);

//---------------------
// MOUSE MOVE function
//---------------------
$("#canvas").mousemove(function(e) {
	if(drawing) {
		var mouseX = e.pageX - this.offsetLeft;
		var mouseY = e.pageY - this.offsetTop;
		addUserGesture(mouseX, mouseY, true);
		drawOnCanvas();
	}
});

//---------------------
// TOUCH MOVE function
//---------------------
canvas.addEventListener("touchmove", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	if(drawing) {
		var rect = canvas.getBoundingClientRect();
		var touch = e.touches[0];

		var mouseX = touch.clientX - rect.left;
		var mouseY = touch.clientY - rect.top;

		addUserGesture(mouseX, mouseY, true);
		drawOnCanvas();
	}
}, false);

//-------------------
// MOUSE UP function
//-------------------
$("#canvas").mouseup(function(e) {
	drawing = false;
});

//---------------------
// TOUCH END function
//---------------------
canvas.addEventListener("touchend", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	drawing = false;
}, false);

//----------------------
// MOUSE LEAVE function
//----------------------
$("#canvas").mouseleave(function(e) {
	drawing = false;
});

//-----------------------
// TOUCH LEAVE function
//-----------------------
canvas.addEventListener("touchleave", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	drawing = false;
}, false);

//--------------------
// ADD CLICK function
//--------------------
function addUserGesture(x, y, dragging) {
	clickX.push(x);
	clickY.push(y);
	clickD.push(dragging);
}

//-------------------
// RE DRAW function
//-------------------
function drawOnCanvas() {
	ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

	ctx.strokeStyle = canvasStrokeStyle;
	ctx.lineJoin    = canvasLineJoin;
	ctx.lineWidth   = canvasLineWidth;

	for (var i = 0; i < clickX.length; i++) {
		ctx.beginPath();
		if(clickD[i] && i) {
			ctx.moveTo(clickX[i-1], clickY[i-1]);
		} else {
			ctx.moveTo(clickX[i]-1, clickY[i]);
		}
		ctx.lineTo(clickX[i], clickY[i]);
		ctx.closePath();
		ctx.stroke();
	}
}

//------------------------
// CLEAR CANVAS function
//------------------------
function clearCanvas(id) {
	ctx.clearRect(0, 0, canvasWidth, canvasHeight);
	clickX = new Array();
	clickY = new Array();
	clickD = new Array();
}

//-------------------------------------
// loader for digitrecognizermlp model
//-------------------------------------
async function loadModel(modelName) {
  console.log("model loading..");

  // clear the model variable
  model = undefined;

  // load the model using a HTTPS request (where you have stored your model files)
  model = await tf.loadLayersModel("https://gogul09.github.io/models/" + modelName + "/model.json");

  console.log("model loaded..");
}

loadModel(modelName);

//-----------------------------------------------
// preprocess the canvas to be MLP friendly
//-----------------------------------------------
function preprocessCanvas(image, modelName) {

	// if model is not available, send the tensor with expanded dimensions
	if (modelName === undefined) {
		alert("No model defined..")
	}

	// if model is digitrecognizermlp, perform all the preprocessing
	else if (modelName === "digitrecognizermlp") {

		// resize the input image to digitrecognizermlp's target size of (784, )
		let tensor = tf.browser.fromPixels(image)
		    .resizeNearestNeighbor([28, 28])
		    .mean(2)
		    .toFloat()
			.reshape([1 , 784]);
		return tensor.div(255.0);
	}

	// if model is digitrecognizercnn, perform all the preprocessing
	else if (modelName === "digitrecognizercnn") {
		// resize the input image to digitrecognizermlp's target size of (1, 28, 28)
		let tensor = tf.browser.fromPixels(image)
		    .resizeNearestNeighbor([28, 28])
		    .mean(2)
		    .expandDims(2)
		    .expandDims()
		    .toFloat();
		console.log(tensor.shape);
		return tensor.div(255.0);
	}

	// else throw an error
	else {
		alert("Unknown model name..")
	}
}


async function predict() {
	var imageData = canvas.toDataURL();
	let tensor = preprocessCanvas(canvas, modelName);
	let predictions = await model.predict(tensor).data();
	let results = Array.from(predictions)
  let chosen = Math.max.apply(Math, results);
  let chosenIndex = results.indexOf(chosen);

  if (chosen > 0.8) {
    document.getElementById("result_box").innerHTML = `I'm pretty sure that you're writing a ${chosenIndex}`;
  } else if (chosen > 0.6) {
    document.getElementById("result_box").innerHTML = `It's just a tad messy but it should be a ${chosenIndex}`;
  } else if (chosen > 0.4) {
    document.getElementById("result_box").innerHTML = `Hmm, it's quite ambiguous but i'm leaning towards a ${chosenIndex}`;
  } else {
    document.getElementById("result_box").innerHTML = `I can't tell what you're writing, i'm just going to go with ${chosenIndex}`;
  }
}
