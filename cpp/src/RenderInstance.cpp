#include "RenderInstance.h"
template class RenderInstance<float>;
template class RenderInstance<double>;

template <typename T>
RenderInstance<T>::RenderInstance(RenderParam params, GLuint solidShader, GLuint fadeShader, LineStrip<T>* vector, Lines<T>* trail, int width, int height)
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
		struct ViewInstance<T>& iview = this->views[i];
		if (iview.valid = pview.valid) {
			//Vertical flip
			iview.viewMtx = mat3<T>(
				T(2.0 * (iview.xScale = std::min(width, height) / params.width) * pview.zoom),
				0,
				0,

				0,
				T(-2.0 * (iview.yScale = std::min(width, height) / params.height) * pview.zoom),
				0,

				T(iview.offsetX = 2.0f * (pview.center_x + double(pview.screen_width - params.width) / 2 + pview.screen_x) / params.width),
				T(iview.offsetY = 2.0f * (pview.center_y + double(pview.screen_height - params.height) / 2 + pview.screen_y) / params.height),
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
			double xy[4] = { 2 * (double(pview.screen_x) / params.width) - 1, 2 * (double(pview.screen_y) / params.height) - 1,
				2 * (double(pview.screen_x + pview.screen_width) / params.width) - 1, 2 * (double(pview.screen_y + pview.screen_height) / params.height) - 1 };
			vec2<T> vertices[4] = {
				vec2<T>(iview.invViewMtx * vec3<T>(xy[0],xy[1],1)),
				vec2<T>(iview.invViewMtx * vec3<T>(xy[0],xy[3],1)),
				vec2<T>(iview.invViewMtx * vec3<T>(xy[2],xy[3],1)),
				vec2<T>(iview.invViewMtx * vec3<T>(xy[2],xy[1],1))
			};
			iview.rect = new Rect<T>((T*)vertices);

			glStencilMask(pview.no_background ? 1 << i : (2 << i) - 1);
			glStencilFunc(GL_ALWAYS, 1 << i, 0xff);
			iview.rect->draw(solidShader, iview.viewMtx, vec2<T>(0), vec3<T>(0), 1, true);

			if (pview.border_on_other_views)
				globalBorders.push_back(&iview);
		}
	}
	glStencilMask(0);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	glStencilFunc(GL_NEVER, 0xff, 0xff);
}

template <typename T>
RenderInstance<T>::~RenderInstance() {
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

template <typename T>
void RenderInstance<T>::draw(const T& time, vec2<T>* pos) {
	multiBuffer->preDraw();
	glViewport(0, 0, this->width, this->height);
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < 8; i++) {
		const struct ViewInstance<T>& view = this->views[i];
		if (!view.valid) continue;

		glStencilFunc(GL_EQUAL, 0xff, 1<<i);

		if (view.drawBackground) {
			glUseProgram(solidShader);
			view.rect->draw(solidShader, view.viewMtx, vec2<T>(0), view.backgroundColor, 1, true);
		}

		vec2<T> offset = view.followPath ? view.zoom * 2.0f * vec2<T>(-pos->x, pos->y) : vec2<T>(0);
		vector->draw(solidShader, view.viewMtx, offset, view.vectorColor, view.vectorWidth);
		GLuint pathShader = view.pathFade ? fadeShader : solidShader;
		glUseProgram(pathShader);
		glutil::glUniform1<T>(glGetUniformLocation(pathShader, "time"), time);
		trail->draw(pathShader, view.viewMtx, offset, view.pathColor, view.pathWidth);

		for (const struct ViewInstance<T>* vi : globalBorders) {
			if (vi->id == i) continue;
			vec2<T> offset = vi->followPath ? view.zoom * vi->zoom * 4.0f * mat2<T>(vi->invViewMtx) * vec2<T>(pos->x, pos->y) : vec2<T>(0);
			vi->rect->draw(solidShader, view.viewMtx, offset, vi->borderColor, vi->borderWidth, false);
		}

		if (view.borderWidth != 0) {
			glUseProgram(solidShader);
			view.rect->draw(solidShader, view.viewMtx, vec2<T>(0), view.borderColor, view.borderWidth, false);
		}
	}
}

template <typename T>
void RenderInstance<T>::postDraw() {
	multiBuffer->postDraw();
}

template <typename T>
void RenderInstance<T>::encode() {
	uint8_t* ptr = multiBuffer->nextPBO();
	if (ptr != nullptr)
	{
		memcpy(frameraw, ptr, framebytes);
		encoder->pushFrame(frameraw);
	}
	else
		std::cout << this->output_name << ": Frame dropped, unable to read data" << std::endl;
}