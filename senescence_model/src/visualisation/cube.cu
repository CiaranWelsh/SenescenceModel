#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <glm/glm.hpp>


#include "header.h"

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
    gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
    gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
}



//KERNEL DEFINITIONS
/** output_navmaps_to_TBO
 * Outputs navmap agent data from FLAME GPU to a 4 component vector used for instancing
 * @param	agents	pedestrian agent list from FLAME GPU
 * @param	data1 four component vector used to output instance data
 * @param	data2 four component vector used to output instance data
 */
__global__ void output_TissueBlock_to_TBO(
        xmachine_memory_agent_list* agents,
        vec4 vbo,
        glm::vec3 centralise){

    //global thread index
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    vbo[index].id = agents->id[index];
    vbo[index].x = agents->x[index] - centralise;
    vbo[index].y = agents->y[index]- centralise;
    vbo[index].z = agents->z[index]- centralise;
    vbo[index].w = agents->damage[index]- centralise;

}


__global__ void output_Fibroblast_agent_to_VBO(
        xmachine_memory_Fibroblast_list* agents,
        glm::vec4* vbo){

    //global thread index
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    vbo[index].id = agents->id[index];
    vbo[index].x = agents->x[index];
    vbo[index].y = agents->y[index];
    vbo[index].z = agents->z[index];
    vbo[index].w = 1.0;
}

extern void generate_cube_instances(
        GLuint* instances_data1_tbo,
        cudaGraphicsResource_t * p_instances_data1_cgr)
{
//    singleIteration();

    //kernals sizes
    int threads_per_tile = 128;
    int tile_size;
    dim3 grid;
    dim3 threads;
    glm::vec3 centralise;


    //pointer
    glm::vec4 *dptr_1;

    if (get_agent_TissueBlock_default_count() > 0) {
        // map OpenGL buffer object for writing from CUDA
        gpuErrchk(cudaGraphicsMapResources(1, p_instances_data1_cgr));
        gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **) &dptr_1, 0, *p_instances_data1_cgr));

        //cuda block size
        tile_size = (int) ceil((float) get_agent_TissueBlock_default_count() / threads_per_tile);
        grid = dim3(tile_size, 1, 1);
        threads = dim3(threads_per_tile, 1, 1);

        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;

        //kernel
        output_TissueBlock_to_TBO << < grid, threads >> > (get_device_TissueBlock_default_agents(), dptr_1, centralise);
        gpuErrchkLaunch();
        // unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, p_instances_data1_cgr));
    }

//    if (get_agent_Fibroblast_Quiescent_count() > 0) {
//        // map OpenGL buffer object for writing from CUDA
//        gpuErrchk(cudaGraphicsMapResources(1, p_instances_data1_cgr));
//        gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **) &dptr_1, 0, *p_instances_data1_cgr));
//
//        //cuda block size
//        tile_size = (int) ceil((float) get_agent_Fibroblast_Quiescent_count() / threads_per_tile);
//        grid = dim3(tile_size, 1, 1);
//        threads = dim3(threads_per_tile, 1, 1);
//
//        centralise = getMaximumBounds() + getMinimumBounds();
//        centralise /= 2;
//
//        //kernel
//        output_Fibroblast_Quiescent_to_TBO << < grid, threads >> > (get_device_TFibroblast_Quiescent_agents(), dptr_1, centralise);
//        gpuErrchkLaunch();
//        // unmap buffer object
//        gpuErrchk(cudaGraphicsUnmapResources(1, p_instances_data1_cgr));
//    }
//
//    if (get_agent_Fibroblast_Repair_count() > 0) {
//        // map OpenGL buffer object for writing from CUDA
//        gpuErrchk(cudaGraphicsMapResources(1, p_instances_data1_cgr));
//        gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **) &dptr_1, 0, *p_instances_data1_cgr));
//
//        //cuda block size
//        tile_size = (int) ceil((float) get_agent_TissueBlock_default_count() / threads_per_tile);
//        grid = dim3(tile_size, 1, 1);
//        threads = dim3(threads_per_tile, 1, 1);
//
//        centralise = getMaximumBounds() + getMinimumBounds();
//        centralise /= 2;
//
//        //kernel
//        output_TissueBlock_to_TBO << < grid, threads >> > (get_device_TissueBlock_default_agents(), dptr_1, centralise);
//        gpuErrchkLaunch();
//        // unmap buffer object
//        gpuErrchk(cudaGraphicsUnmapResources(1, p_instances_data1_cgr));
//    }
}



















