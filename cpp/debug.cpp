#include "core.h"
#include "windows.h"
#include "constant.h"


typedef void (CALLBACK* DLLFUNC_RENDER)(float*, size_t, int, int, float, float, float,
	float, bool, glm::vec3, glm::vec3, int, int, const char*, bool, bool, bool);
int main()
{
	HINSTANCE hDLL = LoadLibrary(".\\render.dll");
	if (hDLL != NULL)
	{
		DLLFUNC_RENDER render = (DLLFUNC_RENDER)GetProcAddress(hDLL, "render");
#define LEN 8
		float data[LEN] = { 0, 0, 0, 50, 20, 0, 0, -50 };
		render(data, LEN, 400, 400, 1 / 60.0, 6.28, 0, 1, false, glm::vec3(1, 1, 0), glm::vec3(1, 1, 1), 60, 1, "out.mp4", true, false, false);
	}
	return 0;
}