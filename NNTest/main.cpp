#include <iostream>
#include <random>
#include <array>
#include <tuple>
#include <string>

float linearCombiner(float a, float b) {
	return a + b;
}

float stepFunc(float input) {
	return int(input >= 0.0f);
}

float signFunc(float input) {
	return int(input >= 0.0f) * 2 - 1;
}

float Yp(float (*func)(float), float x1, float x2, float w1, float w2, float threshold) {
	return func(linearCombiner(x1 * w1, x2 * w2) - threshold);
}

float error(float expectedOutput, float output) {
	return expectedOutput - output;
}

void Wp1(float& w, float learningRate, float error, float& x) {
	w = w + learningRate * x * error;
}

#define MAX_EPOCH 100
#define DATA_COUNT 4
#define ITER_COUNT 4
//perceptron
void perceptron(std::array<std::tuple<float, float, float>, DATA_COUNT>& data, float (*func)(float), float learningRate, float threshold) {

	std::srand(std::time(NULL));
	float x1;
	float x2;
	float expected;

	// Latent space
	float w1 = 0.5f;
	float w2 = 0.5f;
	//float w2 = float(std::rand()) / float(RAND_MAX);
	//float w1 = float(std::rand()) / float(RAND_MAX);

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
		std::tuple<float, float, float>(0.0f,0.0f,0.0f),
		std::tuple<float, float, float>(0.0f,1.0f,0.0f),
		std::tuple<float, float, float>(1.0f,0.0f,0.0f),
		std::tuple<float, float, float>(1.0f,1.0f,1.0f),
	};

	std::array<std::tuple<float, float, float>, DATA_COUNT> XORdata = {};
	//Sign function learning seems to loop weights between same values. Try applying apative learning rate
	std::cout << "Sign Func\n";
	perceptron(ANDdata, signFunc, 0.0001f, 0.2f);
	std::cout << "Step Func\n";
	perceptron(ANDdata, stepFunc, 0.1f, 0.2f);
}