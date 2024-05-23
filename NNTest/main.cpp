#include <iostream>
#include <random>
#include <array>
#include <tuple>
#include <string>
#include <cmath>

float linearCombiner(float a, float b) {
	return a + b;
}

float stepFunc(float input) {
	return int(input >= 0.0f);
}

float signFunc(float input) {
	return int(input >= 0.0f) * 2 - 1;
}

float sigmoid(float input) {
	return 1.0f / (1 + std::exp(-input));
}

float errorGradient(float Yp, float error) {
	return Yp * (1 - Yp) * error;
}

float Yp(float (*func)(float), float x1, float x2, float w1, float w2, float threshold) {
	return func(linearCombiner(x1 * w1, x2 * w2) - threshold);
}

float error(float expectedOutput, float output) {
	return expectedOutput - output;
}

//Calculates new w for p+1
void Wp1(double& w, double learningRate, double error, double& x) {
	w = w + learningRate * x * error;
}

//Calculates new w for p+1
double Wp1_f(double& w, double& learningRate, double& error, double& x) {
	//std::cout << "\tw: " + std::to_string(w) + " lR: " + std::to_string(learningRate) + " E: " + std::to_string(error) + " x: " + std::to_string(x) + "\n";
	//std::cout << "Delta w: " + std::to_string(learningRate * x * error) + "\n";
#ifdef PRINT
#endif
	return w + learningRate * x * error;
}

#define MAX_EPOCH 1000000
#define DATA_COUNT 4
#define ITER_COUNT 4
//#define PRINT

#define PER_LAYER_WEIGHTCOUNT 2
#define NODECOUNT 3

double initializer(double inputCount) {
	return (4.8 * (double(std::rand()) / double(RAND_MAX)) - 2.4) / inputCount;
}

void printNetwork(std::array<std::array<double, PER_LAYER_WEIGHTCOUNT>, NODECOUNT>& weights, std::array<double, NODECOUNT>& thresholds) {

	int nodeI = 0;
	for (std::array<double, PER_LAYER_WEIGHTCOUNT>& weightCol : weights) {

		std::cout << " Node" + std::to_string(nodeI) + "   ";
		int weightI = 0;
		for (double& w : weightCol) {
			std::cout << " W" + std::to_string(weightI) + ": " + std::to_string(w);

			weightI++;
		}
		std::cout << " T: " + std::to_string(thresholds[nodeI]) + "\n";
		nodeI++;
	}
}

//Multi-layer NN with back-propagation
// PER_LAYER_WEIGHTCOUNT != 2 will break this code
void multiLayered(std::array<std::tuple<float, float, float>, DATA_COUNT>& data, double learningRate) {
	std::srand(std::time(NULL));
	constexpr float inputCount = 2.0f;

	/// two input nodes in input layer
	/// two hidden nodes in hidden layer
	/// one output node in output layer
	///	| w[0][0] w[1][0] w[2][0] | 
	/// | w[0][1] w[1][1] w[2][1] |
	/// hidden node 1 wT:	w[0][0] w[0][1]
	/// hidden node	2 wT:	w[1][0] w[1][1]
	/// output node	  wT:	w[2][0] w[2][1]
	/// wT means weights and Thresholds
	std::array<std::array<double, PER_LAYER_WEIGHTCOUNT>, NODECOUNT> weights;
	std::array<double, 3> thresholds;
	// Per-node Yp values
	// y[2] is the output
	std::array<double, 3> y;

	// Randomly initialize all the weights.
	// Does Neuron by Neuron mean weights in same node should be initialized as same values?
	for (std::array<double, PER_LAYER_WEIGHTCOUNT>& weightCol : weights) {
		for (double& w : weightCol) {
			w = initializer(inputCount);
		}
	}
	//Randomly initialize all the thresholds
	for (double& threshold : thresholds) {
		threshold = initializer(inputCount);
	}
	double x1;
	double x2;
	double expected;

	int p = 0;
	int dI = 0;

	bool hasError = true;
	while (hasError)
	{
		double errorSqSum = 0;
		//learningRate = learningRate * 0.8f;
#ifdef PRINT
		std::cout << "EPOCH " + std::to_string(p) + ":\n";
		printNetwork(weights, thresholds);
#endif
		while (dI < DATA_COUNT) {
			x1 = std::get<0>(data[dI]);
			x2 = std::get<1>(data[dI]);
			expected = std::get<2>(data[dI]);
			
			// hidden layer operation
			// terrible code
			//This is why PER_LAYER_WEIGHTCOUNT can not be changed
			for (int i = 0; i < 2;i++) {
				double w1 = weights[i][0];
				double w2 = weights[i][1];
				double threshold = thresholds[i];

				//TODO: rewrite Yp() to accept variable number of inputs/weight pairs
				y[i] = Yp(sigmoid, x1, x2, w1, w2, threshold);
			}
			// output layer operation
			{
				double w1 = weights[2][0];
				double w2 = weights[2][1];
				double threshold = thresholds[2];
				y[2] = Yp(sigmoid, y[0], y[1], w1, w2, threshold);
			}

			double output = y[2];

			double error = expected - output;
#ifdef PRINT
			std::cout << "\t I1: " + std::to_string(x1) + " I2 " + std::to_string(x2) + " O:" + std::to_string(output) + " E: " + std::to_string(error) + "\n";
#endif
			//hasError = abs(error) > 0.01f;
			errorSqSum += abs(error) * abs(error);
			double eG = errorGradient(y[2], error);
			//threshold error adjustment
			thresholds[2] = thresholds[2] + (-1) * learningRate * eG;
			
#ifdef PRINT
#endif
			weights[2][0] = Wp1_f(weights[2][0], learningRate, eG, y[0]);
			weights[2][1] = Wp1_f(weights[2][1], learningRate, eG, y[1]);

			//hidden layer error gradient
			// sum of (error gradient of each node in next layer * weights of each node in next layer)
			double eGH0 = y[0] * (1 - y[0]) * (eG * weights[2][0]);
			double eGH1 = y[1] * (1 - y[1]) * (eG * weights[2][1]);
			//Error gradient is same as the error from the perceptron

			thresholds[0] = thresholds[0] + (-1) * learningRate * eGH0;
			thresholds[1] = thresholds[1] + (-1) * learningRate * eGH1;
			
			//Node 0 weight updates
			weights[0][0] = Wp1_f(weights[0][0], learningRate, eGH0, x1);
			weights[0][1] = Wp1_f(weights[0][1], learningRate, eGH1, x2);
			//Node 1 weight updates
			weights[1][0] = Wp1_f(weights[1][0], learningRate, eGH0, x1);
			weights[1][1] = Wp1_f(weights[1][1], learningRate, eGH1, x2);

			//End of single training. increment to next data
			dI++;
		}
		std::cout << "EPOCH " + std::to_string(p) + " ErrorSquared: " + std::to_string(errorSqSum) + "\n";
		if (errorSqSum < 0.001f) hasError = false;
		dI = 0;
		p++;
	}
	if (p >= MAX_EPOCH) std::cout << "Failed after " + std::to_string(MAX_EPOCH) + " iterations... Adjust learningRate and threshold\n";
	std::cout << "Final result:\n";
	printNetwork(weights, thresholds);
	std::cout << "Output Test:\n";

	{
		for(std::tuple<float, float, float> f : data){
			x1 = std::get<0>(f);
			x2 = std::get<1>(f);
			expected = std::get<2>(f);
			for (int i = 0; i < 2; i++) {
				double w1 = weights[i][0];
				double w2 = weights[i][1];
				double threshold = thresholds[i];

				//TODO: rewrite Yp() to accept variable number of inputs/weight pairs
				y[i] = Yp(sigmoid, x1, x2, w1, w2, threshold);
			}
			// output layer operation
			{
				double w1 = weights[2][0];
				double w2 = weights[2][1];
				double threshold = thresholds[2];
				y[2] = Yp(sigmoid, y[0], y[1], w1, w2, threshold);
			}
			double output = y[2];
			std::cout << "x1: " + std::to_string((int)x1) + " x2: " + std::to_string((int)x2) + " expected: " + std::to_string((int)expected) + "\n";
		}
	}

}

//perceptron
void perceptron(std::array<std::tuple<float, float, float>, DATA_COUNT>& data, float (*func)(float), float learningRate, float threshold) {
	std::srand(std::time(NULL));

	double x1;
	double x2;
	double expected;

	double w1 = 0.5f;
	double w2 = 0.5f;

	int p = 0;
	int dI = 0;

	bool hasError = true;

	while (hasError && p < MAX_EPOCH)
	{
		hasError = false;
		std::cout << "P" + std::to_string(p) + " w1: " + std::to_string(w1) + " w2: " + std::to_string(w2) + "\n";
		while (dI < DATA_COUNT) {

			x1 = std::get<0>(data[dI]);
			x2 = std::get<1>(data[dI]);
			expected = std::get<2>(data[dI]);
		
			float output = Yp(func,x1,x2,w1,w2,threshold);
			float error = expected - output;
			
			std::cout << "\tx1: " + std::to_string(x1) + " x2: " + std::to_string(x2) + " expected: " + std::to_string(expected) + " output: " + std::to_string(output) + "\n";
			std::cout << "\t\tError: " + std::to_string(error) + "\n";

			Wp1(w1, learningRate, error, x1);
			Wp1(w2, learningRate, error, x2);
			if (error != 0.0f) hasError = true;
			dI++;
		}
		dI = 0;
		p++;
	}
	if (p >= MAX_EPOCH) std::cout << "Failed after " + std::to_string(MAX_EPOCH) + " iterations... Adjust learningRate and threshold\n";
	std::cout << "Final w1: " + std::to_string(w1) + " w2: " + std::to_string(w2) + "\n";
}


int main() {
	//Data Set for AND
	std::array<std::tuple<float, float, float>, DATA_COUNT> ANDdata = {
		std::tuple<int, int, int>(0.0f,0.0f,0.0f),
		std::tuple<int, int, int>(0.0f,1.0f,0.0f),
		std::tuple<int, int, int>(1.0f,0.0f,0.0f),
		std::tuple<int, int, int>(1.0f,1.0f,1.0f)
	};

	std::array<std::tuple<float, float, float>, DATA_COUNT> XORdata = {
		std::tuple<int, int, int>(0.0f,0.0f,0.0f),
		std::tuple<int, int, int>(0.0f,1.0f,1.0f),
		std::tuple<int, int, int>(1.0f,0.0f,1.0f),
		std::tuple<int, int, int>(1.0f,1.0f,0.0f),
	};
	//Sign function learning seems to loop weights between same values. Try applying apative learning rate
	//std::cout << "Sign Func\n";
	//perceptron(ANDdata, signFunc, 0.0001f, 0.2f);
	//std::cout << "Step Func\n";
	//perceptron(ANDdata, stepFunc, 0.1f, 0.2f);
	multiLayered(XORdata, 0.1f);
}