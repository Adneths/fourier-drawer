#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <cstdlib>

#include "core.h"
#include "Shader.h"
#include "LineStrip.h"
#include <complex>
#include <string>

#include "VideoEncoder.h"
#include "FourierSeries.h"
#include "NpFourierSeries.h"
#include "MultiBuffer.h"

#include "NumCpp.hpp"

#define PI 3.141592

//https://stackoverflow.com/a/26221725
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
	auto size = static_cast<size_t>(size_s);
	std::unique_ptr<char[]> buf(new char[size]);
	std::snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

std::string formatTime(int seconds)
{
	if (seconds > 3599)
		return string_format("%d:%02d:%02d", seconds / 3600, (seconds % 3600) / 60, seconds % 60);
	return string_format("%02d:%02d", seconds / 60, seconds % 60);
}
std::string formatTime(double seconds)
{
	int sec = (int)seconds;
	if (sec > 3599)
		return string_format("%d:%02d:%02d.%03d", sec / 3600, (sec % 3600) / 60, sec % 60, (int)((seconds - sec) * 1000));
	return string_format("%02d:%02d.%03d", sec / 60, sec % 60, (int)((seconds - sec) * 1000));
}

bool alive = true;
void keyboard_interrupt(int signum) {
	alive = false;
}

char buf[256];
int printProgressBar(float part, int barLength = 40, int minLength = 0, std::string prefix = "", std::string suffix = "")
{
	std::string bar = std::string((int)(barLength * part), '*');
	bar += std::string(barLength - bar.length(), ' ');
	int ret = sprintf(buf, "%s |%s| %.1f%% %s", prefix.c_str(), bar.c_str(), part * 100, suffix.c_str());
	printf("%-*s\r", minLength, buf);
	return ret;
}

void GLAPIENTRY errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		type, severity, message);
}

extern "C" {
	__declspec(dllexport) int __cdecl render(float* data, size_t size, int width, int height, float dt, float duration, float start, float trailLength, bool trailFade, glm::vec3 trailColor, glm::vec3 vectorColor, int fps, int fpf, const char* output, bool show, bool debug)
	{
		signal(SIGINT, keyboard_interrupt);

		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW" << std::endl;
			return -1;
		}

		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* window = glfwCreateWindow(width, height, "Fourier", NULL, NULL);
		if (!window) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			return -1;
		}

		glfwMakeContextCurrent(window);
		glewInit();
		glfwSwapInterval(1);

		if (debug)
		{
			glEnable(GL_DEBUG_OUTPUT);
			glDebugMessageCallback(errorCallback, 0);
		}

		GLuint vectorShader = LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/solid.frag", debug);
		GLuint pathShader = trailFade ? LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/fade.frag", debug) : vectorShader;
		if (!vectorShader) {
			std::cerr << "Failed to initialize shader program" << std::endl;
			glfwDestroyWindow(window);
			glfwTerminate();
			return -1;
		}

		//Vertical flip
		glm::mat3 viewMtx = glm::mat3(2.0f/width, 0, 0, 0, -2.0f/height, 0, 0, 0, 1);
		size_t vectorSize = size / 2;
		size_t trailSize = (size_t)(trailLength / dt);

		if (trailFade)
		{
			glUseProgram(pathShader);
			glUniform1f(glGetUniformLocation(pathShader, "trailLength"), trailLength);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		nc::NdArray<std::complex<float>> vecs((std::complex<float>*)data, vectorSize);
		vecs = nc::append(nc::zeros<std::complex<float>>(1,1), vecs);

		LineStrip* vector = new LineStrip(glm::vec2(0,0), vectorSize, vectorColor);
		Lines* trail = new Lines(glm::vec2(0,0), trailSize, trailColor, trailFade);
		
		VideoEncoder* encoder = new VideoEncoder(output, width, height, fps);
		encoder->initialize();

		FourierSeries* fourier = new NpForuierSeries(vector, trail, (std::complex<float>*)data, vectorSize, dt, fpf);
		MultiBuffer* multiBuffer = new MultiBuffer(width, height, 2);

		glClearColor(0, 0, 0, 1);
		float t = start;
		float end = start + duration;
		uint8_t* frameraw = (uint8_t*)malloc(sizeof(uint8_t) * width * height * 3);

		std::string ETR = "XX:XX";
		int ind = 0;
		double sTime = glfwGetTime();
		double pTime = glfwGetTime();
		double d64[64] = {0};
		int len = 0;
		while (t < end && alive) {
			//glfwPollEvents();
			//glfwSwapBuffers(window);

			double time = glfwGetTime();
			glClear(GL_COLOR_BUFFER_BIT);
			vector->draw(vectorShader, viewMtx);
			trail->draw(pathShader, viewMtx, t);
			multiBuffer->nextVBO();

			t += fourier->increment(fpf, t);
			fourier->updateBuffers();

			uint8_t* ptr = multiBuffer->nextPBO();
			if (ptr != nullptr)
			{
				memcpy(frameraw, ptr, width * height * 3);
				encoder->pushFrame(frameraw);
			}
			else
				std::cout << std::endl << "Frame dropped, unable to read data" << std::endl;

			d64[(ind = (ind + 1) & 0b111111)] = glfwGetTime() - time;
			if (glfwGetTime() - pTime > 2)
			{
				pTime = glfwGetTime();
				double sum = 0;
				for (double d : d64)
					sum += d;
				sum /= 64;
				ETR = formatTime((int)(sum * (end - t) / dt)) + " remaining";
			}

			len = printProgressBar((t - start) / duration, 40, len, "Rendering", ETR);
		}
		if (alive)
			std::cout << std::endl << "Total Time: " << formatTime(glfwGetTime() - sTime) << std::endl;
		else
			std::cout << std::endl << "Program Terminated" << std::endl;
		free(frameraw);
		encoder->close();

		delete vector;
		delete trail;
		delete encoder;
		delete fourier;
		delete multiBuffer;

		glDeleteProgram(vectorShader);
		if (trailFade)
			glDeleteProgram(pathShader);

		glfwDestroyWindow(window);
		glfwTerminate();
		return 0;
	}
}