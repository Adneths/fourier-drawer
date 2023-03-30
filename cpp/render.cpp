#include <iostream>

#include "core.h"

extern "C" {
	__declspec(dllexport) int __cdecl render(int width, int height)
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

		while (!glfwWindowShouldClose(window)) {
			glfwSwapBuffers(window);
		}

		glfwDestroyWindow(window);
		glfwTerminate();
	}
}