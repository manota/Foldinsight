from modeller import *
import sys
env = environ()
log.verbose()
aln = alignment(env)
env.io.atom_files_directory = './'
env.io.hetatm = True


def sc_align(template_pdb_path,target_pdb_path):
    for k in [template_pdb_path, target_pdb_path]:
        m = model(env, file=k)
        aln.append_model(m, atom_files=k, align_codes=k)

    #Doing Alignment
    for (weights, write_fit, whole) in (((1., 0., 0., 0., 1., 0.), False, True),
                                        ((1., 0.5, 1., 1., 1., 0.), False, True),
                                        ((1., 1., 1., 1., 1., 0.), True, True)):
        aln.salign(rms_cutoff=3.5, normalize_pp_scores=False, rr_file='$(LIB)/as1.sim.mat', overhang=30,
                    gap_penalties_1d=(-450, -50), gap_penalties_3d=(0, 3), gap_gap_score=0, gap_residue_score=0, alignment_type='tree',
                    feature_weights=weights, improve_alignment=True, fit=True, write_fit=write_fit, write_whole_pdb=False, output='ALIGNMENT QUALITY')

def sc_align_all(pdb_pathes_list):
    for k in pdb_pathes_list:
        m = model(env, file=k)
        aln.append_model(m, atom_files=k, align_codes=k)

    #Doing Alignment
    for (weights, write_fit, whole) in (((1., 0., 0., 0., 1., 0.), False, True),
                                        ((1., 0.5, 1., 1., 1., 0.), False, True),
                                        ((1., 1., 1., 1., 1., 0.), True, True)):
        aln.salign(rms_cutoff=3.5, normalize_pp_scores=False, rr_file='$(LIB)/as1.sim.mat', overhang=30,
                   gap_penalties_1d=(-450, -50), gap_penalties_3d=(0, 3), gap_gap_score=0, gap_residue_score=0, alignment_type='tree',
                   feature_weights=weights, improve_alignment=True, fit=True, write_fit=write_fit, write_whole_pdb=False, output='ALIGNMENT QUALITY')
