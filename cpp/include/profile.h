#define RED     0
#define YELLOW  1
#define GREEN   2
#define AQUA    3
#define BLUE    4
#define PURPLE  5
#define BLACK   6
#define GRAY    7
#define WHITE   8

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
const uint32_t colors[] = { 0xffff0000, 0xffffff00, 0xff00ff00, 0xff00ffff, 0xff0000ff, 0xffff00ff, 0xff000000, 0xff808080, 0xffffffff };
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define POP_RANGE() nvtxRangePop();
#define NAME_THREAD(name) nvtxNameOsThread(pthread_self(), name);
#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define MEASURE(result) for(double startT = -glfwGetTime(); startT <= 0; result += startT = glfwGetTime() + startT)
#else
#define POP_RANGE()
#define NAME_THREAD(name)
#define PUSH_RANGE(name,cid)
#define MEASURE(result)
#endif
