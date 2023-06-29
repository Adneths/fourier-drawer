#include "RenderInstance.h"


RenderInstance::RenderInstance(RenderParam params, GLuint vectorShader, GLuint pathShader, LineStrip* vector, Lines* trail, float width, float height)
	:params(params), vectorShader(vectorShader), pathShader(pathShader), vector(vector), trail(trail) {
	//Vertical flip
	viewMtx = glm::mat3(2.0f * std::min(width, height) / params.width * params.zoom, 0, 0,
		0, -2.0f * std::min(width, height) / params.height * params.zoom, 0,
		offsetX = 2.0f * params.x / width, offsetY = -2.0f * params.y / height, 1.0f);

	multiBuffer = new MultiBuffer(params.width, params.height, 2);
	encoder = new VideoEncoder(params.output, params.width, params.height, params.fps);
	encoder->initialize();

	framebytes = params.width * params.height * 3;
	frameraw = (uint8_t*)malloc(sizeof(uint8_t) * framebytes);
}

RenderInstance::~RenderInstance() {
	free(frameraw);
	encoder->close();
	delete multiBuffer;
	delete encoder;
}

GLsync RenderInstance::draw(const float& time, glm::vec2* pos) {
	multiBuffer->preDraw();

	if (pos != nullptr)
	{
		viewMtx[2][0] = offsetX - 2.0f * params.zoom * pos->x;
		viewMtx[2][1] = offsetY + 2.0f * params.zoom * pos->y;
	}

	glViewport(0, 0, params.width, params.height);
	glClear(GL_COLOR_BUFFER_BIT);
	glUseProgram(vectorShader);
	glUniform3fv(glGetUniformLocation(vectorShader, "DiffuseColor"), 1, (float*)&params.vectorColor);
	glLineWidth(params.vectorWidth);
	vector->draw(vectorShader, viewMtx);
	glUseProgram(pathShader);
	glUniform3fv(glGetUniformLocation(pathShader, "DiffuseColor"), 1, (float*)&params.trailColor);
	glUniform1f(glGetUniformLocation(pathShader, "time"), time);
	glLineWidth(params.trailWidth);
	trail->draw(pathShader, viewMtx);
	GLsync draw = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

	multiBuffer->postDraw();

	return draw;
}

/*void RenderInstance::copy() {
	multiBuffer->nextVBO();
}*/

void RenderInstance::encode() {
	uint8_t* ptr = multiBuffer->nextPBO();
	if (ptr != nullptr)
	{
		memcpy(frameraw, ptr, framebytes);
		encoder->pushFrame(frameraw);
	}
	else
		std::cout << std::endl << params.output << ": Frame dropped, unable to read data" << std::endl;
}