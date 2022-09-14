with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/cnn_dm_pegasus/generated/orig.txt') as orig, \
    open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/cnn_dm_pegasus/generated/beams_5.csv') as beam, \
    open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/cnn_dm_pegasus/generated/eabs_beams_5_width_5_alpha_5.csv') as eabs, \
    open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/cnn_dm_pegasus/generated/all.csv', 'w') as output:

    orig_data = orig.readlines()
    beam_data = beam.readlines()
    eabs_data = eabs.readlines()

    print("Document\tOrignal Summary\tBeam Summary\tEABS Summary", file=output, flush=True)
    for i, line in enumerate(orig_data):
        document, orig_summary = line.strip().split('\t')
        beam_document, beam_summary = beam_data[i].strip().split('\t')
        eabs_document, eabs_summary = eabs_data[i].strip().split('\t')

        # if document == beam_document:
            # import pdb; pdb.set_trace()

        # if document == eabs_document

        print(f"{document.strip()}\t{orig_summary.strip()}\t{beam_summary.strip()}\t{eabs_summary.strip()}", file=output)
