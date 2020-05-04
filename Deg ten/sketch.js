
var x_vals = [];
var y_vals = [];

let a, b, c, d, e, f, g, h, i, j, k;

const learningRate = 0.2;
// const optimizer = tf.train.sgd(learningRate);
const optimizer = tf.train.adam(learningRate);

function setup() {
    createCanvas(window.innerWidth, window.innerHeight);
    // createCanvas(400, 400);
    
    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));
    e = tf.variable(tf.scalar(random(-1, 1)));
    f = tf.variable(tf.scalar(random(-1, 1)));
    g = tf.variable(tf.scalar(random(-1, 1)));
    h = tf.variable(tf.scalar(random(-1, 1)));
    i = tf.variable(tf.scalar(random(-1, 1)));
    j = tf.variable(tf.scalar(random(-1, 1)));
    k = tf.variable(tf.scalar(random(-1, 1)));
    
}

function draw() {

    tf.tidy(() => {
        if(x_vals.length > 0){
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
            // optimizer.minimize(function() {
            //     loss(predict(x_vals), ys);
            // });
        }
    });

    background(0);
    stroke(255);
    strokeWeight(8);
    for(let i =0;i<x_vals.length;i++){
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], 1, -1, 0, height);
        point(px, py);
    }

    const curveX = [];
    for(let x = -1; x < 1; x+=0.05){
        curveX.push(x);
    }
    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for(let i = 0;i< curveX.length; i++){
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], 1, -1, 0, height);
        vertex(x, y);
    }
    endShape();

    console.log(tf.memory().numTensors);
    // noLoop();
}

function mouseDragged() {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

    x_vals.push(x);
    y_vals.push(y);
}
// function mousePressed() {
//     let x = map(mouseX, 0, width, -1, 1);
//     let y = map(mouseY, 0, height, 1, -1);

//     x_vals.push(x);
//     y_vals.push(y);
// }

function predict(x) {
    const xs = tf.tensor1d(x);
    // y = a*x^2 + b*x + c
    // const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
    const ys = xs.pow(tf.scalar(10)).mul(a)
                .add(xs.pow(tf.scalar(9)).mul(b))
                .add(xs.pow(tf.scalar(8)).mul(c))
                .add(xs.pow(tf.scalar(7)).mul(d))
                .add(xs.pow(tf.scalar(6)).mul(e))
                .add(xs.pow(tf.scalar(5)).mul(f))
                .add(xs.pow(tf.scalar(4)).mul(g))
                .add(xs.pow(tf.scalar(3)).mul(h))
                .add(xs.square().mul(i))
                .add(xs.mul(j))
                .add(k);
    return ys;
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}































// function setup() {
// 	noCanvas();
// 	frameRate(1);
// }

// function draw() {
// 	background(0);
// 	const values = [];
// 	for(let i = 0; i < 15; i++) {
// 		values[i] = random(0, 100);
// 	}

// 	const shape = [5, 3];

// 	function my_valstuff() {
// 		const a = tf.tensor(values, shape, 'int32');
// 		const b = tf.tensor(values, shape, 'int32');
// 		const b_t = b.transpose();
// 		const c = a.matMul(b_t);
// 		console.log("hello");
// 	}

// 	console.log(tf.memory().numTensors);

// 	   dispose tensor
// 	   a.dispose();
// 	   b.dispose();
// 	   c.dispose();
// 	   b_t.dispose();
// 	   console.log(tf.memory().numTensors);

// 	   tidy fun
// 	tf.tidy(my_valstuff);
// 	console.log(tf.memory().numTensors);

// 	   const va = tf.variable(a);

// 	   const a = tf.tensor([0, 0, 127, 255, 100, 50, 24, 54], [2, 2, 2], 'int32');

// 	   printing the whole tensor object
// 	   a.print();
// 	   b.print();
// 	   c.print();
// 	   console.log(a.toString());

// 	   printing the array of data
// 	   a.data().then((stuff) => {
// 	   	console.log(stuff);
// 	   });


// 	  using dataSync()
// 	   console.log(a.dataSync());

// 	   get fun to get a particular element
// 	   console.log(a.get(2));

// 	   console.log(va);

	

// }