#ifndef DYNEARTHSOL3D_REMESHING_HPP
#define DYNEARTHSOL3D_REMESHING_HPP

int bad_mesh_quality(const Param&, const Variables&, int&);
void remesh(const Param&, Variables&, int);

void remesh_gpu(const Param&, Variables&, int);
int bad_mesh_quality_gpu(const Param&, const Variables&);
void gpu_download_remeshing(Variables &var);
void gpu_upload_remeshing(Variables &var);

#endif
