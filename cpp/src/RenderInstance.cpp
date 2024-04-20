#include "RenderInstance.h"


RenderInstance::RenderInstance(RenderParam params, GLuint solidShader, GLuint fadeShader, LineStrip* vector, Lines* trail, float width, float height)
	: output_name(params.output_name), width(params.width), height(params.height), views(),
		solidShader(solidShader), fadeShader(fadeShader), vector(vector), trail(trail) {
	multiBuffer = new MultiBuffer(params.width, params.height, 2);
	encoder = new VideoEncoder(params.output_name, params.width, params.height, params.fps);
	encoder->initialize();
	framebytes = params.width * params.height * 3;
	frameraw = (uint8_t*)malloc(sizeof(uint8_t) * framebytes);

	glViewport(0, 0, params.width, params.height);
	multiBuffer->preDraw();
	glClearStencil(0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
	glStencilFunc(GL_ALWAYS, 0xff, 0xff);
	glStencilMask(0xff);
	for (int i = 0; i < 8; i++) {
		const struct RenderParam::View& pview = params.views[i];
		struct ViewInstance& iview = this->views[i];
		if (iview.valid = pview.valid) {
			//Vertical flip
			iview.viewMtx = glm::mat3(
				2.0f * (iview.xScale = std::min(width, height) / params.width) * pview.zoom,
				0,
				0,

				0,
				-2.0f * (iview.yScale = std::min(width, height) / params.height) * pview.zoom,
				0,

				iview.offsetX = 2.0f * (pview.center_x + float(pview.screen_width - params.width) / 2 + pview.screen_x) / params.width,
				iview.offsetY = 2.0f * (pview.center_y + float(pview.screen_height - params.height) / 2 + pview.screen_y) / params.height,
				1.0f
			);

			iview.id = i;
			iview.zoom = pview.zoom;
			iview.vectorColor = pview.vector_color;
			iview.pathColor = pview.path_color;
			iview.backgroundColor = pview.background_color;
			iview.borderColor = pview.border_color;
			iview.vectorWidth = pview.vector_width;
			iview.pathWidth = pview.path_width;
			iview.borderWidth = pview.border_width;

			iview.drawBackground = !pview.no_background;
			iview.drawBorderOnOther = pview.border_on_other_views;
			iview.followPath = pview.follow_path;
			iview.pathFade = pview.path_fade;

			iview.invViewMtx = glm::inverse(iview.viewMtx);
			float xy[4] = { 2 * (float(pview.screen_x) / params.width) - 1, 2 * (float(pview.screen_y) / params.height) - 1,
				2 * (float(pview.screen_x + pview.screen_width) / params.width) - 1, 2 * (float(pview.screen_y + pview.screen_height) / params.height) - 1 };
			glm::vec2 vertices[4] = {
				glm::vec2(iview.invViewMtx * glm::vec3(xy[0],xy[1],1)),
				glm::vec2(iview.invViewMtx * glm::vec3(xy[0],xy[3],1)),
				glm::vec2(iview.invViewMtx * glm::vec3(xy[2],xy[3],1)),
				glm::vec2(iview.invViewMtx * glm::vec3(xy[2],xy[1],1))
			};
			iview.rect = new Rect((float*)vertices);

			glStencilMask(1 << i);
			iview.rect->draw(solidShader, iview.viewMtx, glm::vec2(0), glm::vec3(0), 1, true);

			if (pview.border_on_other_views)
				globalBorders.push_back(&iview);
		}
	}
	glStencilMask(0);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	glStencilFunc(GL_NEVER, 0xff, 0xff);
}

RenderInstance::~RenderInstance() {
	free(frameraw);
	encoder->close();
	delete multiBuffer;
	delete encoder;
	for (int i = 0; i < 8; i++) {
		if (this->views[i].valid) {
			delete this->views[i].rect;
		}
	}
}

void RenderInstance::draw(const float& time, glm::vec2* pos) {
	multiBuffer->preDraw();
	glViewport(0, 0, this->width, this->height);
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < 8; i++) {
		const struct ViewInstance& view = this->views[i];
		if (!view.valid) continue;

		glStencilFunc(GL_EQUAL, 0xff, 1<<i);

		if (view.drawBackground) {
			glUseProgram(solidShader);
			view.rect->draw(solidShader, view.viewMtx, glm::vec2(0), view.backgroundColor, 1, true);
		}

		glm::vec2 offset = view.followPath ? view.zoom * 2.0f * glm::vec2(-pos->x, pos->y) : glm::vec2(0);
		vector->draw(solidShader, view.viewMtx, offset, view.vectorColor, view.vectorWidth);
		GLuint pathShader = view.pathFade ? fadeShader : solidShader;
		glUseProgram(pathShader);
		glUniform1f(glGetUniformLocation(pathShader, "time"), time);
		trail->draw(pathShader, view.viewMtx, offset, view.pathColor, view.pathWidth);

		for (const struct ViewInstance* vi : globalBorders) {
			if (vi->id == i) continue;
			glm::vec2 offset = vi->followPath ? view.zoom * vi->zoom * 4.0f * glm::mat2(vi->invViewMtx) * glm::vec2(pos->x, pos->y) : glm::vec2(0);
			vi->rect->draw(solidShader, view.viewMtx, offset, vi->borderColor, vi->borderWidth, false);
		}

		if (view.borderWidth != 0) {
			glUseProgram(solidShader);
			view.rect->draw(solidShader, view.viewMtx, glm::vec2(0), view.borderColor, view.borderWidth, false);
		}
	}
}

void RenderInstance::postDraw() {
	multiBuffer->postDraw();
}

void RenderInstance::encode() {
	uint8_t* ptr = multiBuffer->nextPBO();
	if (ptr != nullptr)
	{
		memcpy(frameraw, ptr, framebytes);
		encoder->pushFrame(frameraw);
	}
	else
		std::cout << this->output_name << ": Frame dropped, unable to read data" << std::endl;
}