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

#define NAME_THREAD(name) nvtxNameOsThread(pthread_self(), name);
#define POP_RANGE() nvtxRangePop();
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
#define MAKE_RANGE(rid) nvtxRangeId_t rid;
#define END_RANGE(rid) nvtxRangeEnd(rid);
#define START_RANGE(rid,name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    rid = nvtxRangeStartEx(&eventAttrib); \
}
#define MEASURE(result) for(double startT = -glfwGetTime(); startT <= 0; result += startT = glfwGetTime() + startT)
#else
#define NAME_THREAD(name)
#define POP_RANGE()
#define PUSH_RANGE(name,cid)
#define MAKE_RANGE(rid)
#define END_RANGE(rid)
#define START_RANGE(rid,name,cid)
#define MEASURE(result)
#endif

#include <thread>
#ifdef _WIN32
#include <windows.h>
const DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType; // Must be 0x1000.
    LPCSTR szName; // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)


void SetThreadName(uint32_t dwThreadID, const char* threadName)
{

    // DWORD dwThreadID = ::GetThreadId( static_cast<HANDLE>( t.native_handle() ) );

    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
    }
}
void SetThreadName(const char* threadName)
{
    SetThreadName(GetCurrentThreadId(), threadName);
}

void SetThreadName(std::thread * thread, const char* threadName)
{
    DWORD threadId = ::GetThreadId(static_cast<HANDLE>(thread->native_handle()));
    SetThreadName(threadId, threadName);
}

#elif defined(__linux__)
#include <sys/prctl.h>
void SetThreadName(const char* threadName)
{
    prctl(PR_SET_NAME, threadName, 0, 0, 0);
}

#else
void SetThreadName(std::thread * thread, const char* threadName)
{
    auto handle = thread->native_handle();
    pthread_setname_np(handle, threadName);
}
#endif
