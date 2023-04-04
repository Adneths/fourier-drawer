#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "Shader.h"
#include "LineStrip.h"
#include <complex>

#include "VideoEncoder.h"
#include "FourierSeries.h"
#include "NpFourierSeries.h"

#include "NumCpp.hpp"

#define PI 3.141592

extern "C" {
	__declspec(dllexport) int __cdecl render(float* data, size_t size, int width, int height, float dt, float duration, float start, float trailLength, glm::vec3 trailColor, glm::vec3 vectorColor, int fps, int fpf, const char* output)
	{
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW" << std::endl;
			return -1;
		}

		GLFWwindow* window = glfwCreateWindow(width, height, "Fourier", NULL, NULL);
		if (!window) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			return -1;
		}

		glfwMakeContextCurrent(window);
		glewInit();
		glfwSwapInterval(1);

		GLuint vectorShader = LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/solid.frag");
		if (!vectorShader) {
			std::cerr << "Failed to initialize shader program" << std::endl;
			glfwDestroyWindow(window);
			glfwTerminate();
			return -1;
		}


		//Vertical flip
		glm::mat3 viewMtx = glm::mat3(2.0f/width, 0, 0, 0, -2.0f/height, 0, 0, 0, 1);
		size_t vectorSize = size / 2;
		size_t trailSize = (int)(2 * PI * 60 / trailLength);

		nc::NdArray<std::complex<float>> vecs((std::complex<float>*)data, vectorSize);
		vecs = nc::append(nc::zeros<std::complex<float>>(1,1), vecs);

		LineStrip* vector = new LineStrip(glm::vec2(0,0), vectorSize+1, vectorColor);
		CyclicalLineStrip* trail = new CyclicalLineStrip(glm::vec2(-300,-200), trailSize, trailColor);

		VideoEncoder* encoder = new VideoEncoder(output, width, height, fps);
		encoder->initialize();

		FourierSeries* fourier = new NpForuierSeries(vector, trail, (std::complex<float>*)data, vectorSize, dt, fpf);

		glClearColor(0, 0, 0, 1);
		float t = start;
		float end = start + duration;
		uint8_t* frameraw = (uint8_t*)malloc(sizeof(uint8_t) * width * height * 3);
		while (t < end) {
			glfwPollEvents();
			glfwSwapBuffers(window);

			glClear(GL_COLOR_BUFFER_BIT);
			vector->draw(vectorShader, viewMtx);
			//trail->draw(vectorShader, viewMtx);

			t += fourier->increment(fpf);
			fourier->updateBuffers();

			glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameraw);
			encoder->pushFrame(frameraw);
		}
		free(frameraw);
		encoder->close();

		delete vector;
		delete trail;
		delete encoder;
		delete fourier;

		glfwDestroyWindow(window);
		glfwTerminate();
		return 0;
	}
}