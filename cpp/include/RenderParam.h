#pragma once

#include "core.h"
#include <iostream>

struct RenderParam {
	char* output_name;
	int width, height, fps;
	struct View {
		bool valid = false;
		bool no_background;
		glm::vec3 background_color, border_color;
		float border_width;
		bool border_on_other_views;

		int screen_x, screen_y, screen_width, screen_height;
		float center_x, center_y, zoom;

		float vector_width, path_width;
		glm::vec3 vector_color, path_color;
		bool follow_path, path_fade;
	} views[8];
};

inline std::ostream& operator<<(std::ostream& os, const RenderParam& p) {
	os << "{ output_name=" << p.output_name << ", width=" << p.width << ", height=" << p.height << ", fps=" << p.fps;
	os << ", views=[ ";
	for (size_t i = 0; i < 8; i++) if (p.views[i].valid) {
		if (i != 0) os << ", ";
		os << "{ no_background=" << p.views[i].no_background << ", background_color = (" << p.views[i].background_color.x << ", " << p.views[i].background_color.y << ")";
		os << ", border_color=(" << p.views[i].border_color.x << ", " << p.views[i].border_color.y << ", " << p.views[i].border_color.z << ")";
		os << ", border_width=" << p.views[i].border_width << ", border_on_other_views=" << p.views[i].border_on_other_views;
		os << ", screen_x=" << p.views[i].screen_x << ", screen_y=" << p.views[i].screen_y;
		os << ", screen_width=" << p.views[i].screen_width << ", screen_height=" << p.views[i].screen_height;
		os << ", center_x=" << p.views[i].center_x << ", center_y=" << p.views[i].center_y;
		os << ", zoom=" << p.views[i].zoom;
		os << ", vector_width=" << p.views[i].vector_width << ", path_width=" << p.views[i].path_width;
		os << ", vector_color=(" << p.views[i].vector_color.x << ", " << p.views[i].vector_color.y << ", " << p.views[i].vector_color.z << ")";
		os << ", path_color=(" << p.views[i].path_color.x << ", " << p.views[i].path_color.y << ", " << p.views[i].path_color.z << ")";
		os << ", follow_path=" << p.views[i].follow_path << ", path_fade=" << p.views[i].path_fade << " }";
	}
	return os << " ] }";
}