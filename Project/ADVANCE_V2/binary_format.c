/** 
* @author Navneet Dalal (Navneet.Dalal@inrialpes.fr)
* Support for binary data format added to SVM Light, as it takes too 
* long to read supported text format.
*/
#include "svm_common.h"
#include <stdio.h>

long verbosity;
//#ifdef __cplusplus
//extern "C" {
//#endif
//int    isnan(double);
//void   *my_malloc(size_t);
//long   maxl(long, long);
//#ifdef __cplusplus
//}
//#endif

//extern "C" int    isnan(double);
//extern "C" void   *my_malloc(size_t);
//extern "C" long   maxl(long, long);

void typeid_verbose(int type) {
	switch (type) {
	case 3:
		printf("Will read 'int' type target values\n");
		break;
	case 4:
		printf("Will read 'float' type target values\n");
		break;
	case 5:
		printf("Will read 'double' type target values\n");
		break;
	default:
		printf("Default type-id. Will read double type "
			"target values\n");
		break;
	};
}
void read_binary_documents(const char *docfile, DOC ***docs, double **label, 
	long int *totwords, long int *totdoc, long* verb)
{// {{{
	char *comment;
	WRD *words;
	long dnum=0,wpos,dpos=0,dneg=0,dunlab=0,queryid,slackid,max_docs;
	long max_words_doc;
	double costfactor;
	double doc_label;
	FILE *docfl;
	int data_typeid = 0;
	int target_typeid = 0;
	int feature_length = 0, num_feature = 0;
	char msg[512];
	verbosity = (*verb);

	if ((docfl = fopen (docfile, "rb")) == NULL)
	{ perror (docfile); exit (1); }

	if (!fread (&data_typeid,sizeof(int),1,docfl))
	{ perror ("Unable to read data type id"); exit (1); }
	if (!fread (&target_typeid,sizeof(int),1,docfl))
	{ perror ("Unable to read target type id"); exit (1); }
	
	if(verbosity>=1) {
		typeid_verbose(data_typeid);
		typeid_verbose(target_typeid);
	}

	/* scan size of input file */
	if (!fread (&num_feature,sizeof(int),1,docfl))
	{ perror ("Unable to read number of feature"); exit (1); }

	if (!fread (&feature_length,sizeof(int),1,docfl))
	{ perror ("Unable to read feature vector length"); exit (1); }

	max_words_doc = feature_length; max_docs = num_feature;
	(*totwords)=max_words_doc;
	if(verbosity>=1) {
		printf("Feature length %d, Feature count %d\n",
			feature_length,num_feature);
	}

	if((*totwords) > MAXFEATNUM) {
		printf("\nMaximum feature number exceeds limit defined in MAXFEATNUM!\n");
		exit(1);
	}
	/* set comment to something for time being */
	comment = (char*) my_malloc(sizeof(char)*1);
	*comment = 0;

	(*docs) = (DOC **)my_malloc(sizeof(DOC *)*max_docs);    /* feature vectors */
	(*label) = (double *)my_malloc(sizeof(double)*max_docs); /* target values */

	words = (WRD *)my_malloc(sizeof(WRD)*(max_words_doc+1));
	dnum=0;

	if(verbosity>=2) {
		printf("Reading examples into memory..."); fflush(stdout);
	}
	while(!feof(docfl) && dnum < max_docs) {
		/* wpos contains type id for time being*/
		if(!read_feature(docfl,words,&doc_label,
			target_typeid, data_typeid,
			&queryid,&slackid,&costfactor,
			&wpos,max_words_doc,&comment)) {
				printf("\nParsing error in vector %ld!\n",dnum);
				exit(1);
		}
		(*label)[dnum]=doc_label;
		if (doc_label < 0) dneg++;
		else if(doc_label > 0) dpos++;
		else if (doc_label == 0) dunlab++;
		(*docs)[dnum] = create_example(dnum,queryid,slackid,costfactor,
			create_svector(words,*totwords,comment,1.0));
		dnum++;  
		if(verbosity>=3 && ((dnum % 100) == 0)) {
			printf("%ld..",dnum); fflush(stdout);
		}
	} 

	fclose(docfl);
	free(words);
	free(comment);
	if(verbosity>=1) {
		fprintf(stdout, "OK. (%ld examples read)\n", dnum);
	}
	(*totdoc)=dnum;
}// }}}
int read_feature(FILE *docfl, WRD *words, double *label,
	int target_typeid, int data_typeid,
	long *queryid, long *slackid, double *costfactor,
	long int *numwords, long int max_words_doc,
	char **comment)
{// {{{
	register long wpos;

	(*queryid)=0;
	(*slackid)=0;
	(*costfactor)=1;
	/* do not modify comment for time being*/
	/* (*comment)=NULL; */

	if (feof(docfl) == EOF) {
		perror("premature EOF");
		return 0;
	} else if (ferror(docfl)) {
		perror("Unexpected error, unable to read all features");
		return 0;
	}

	/* read the target value */
	switch (target_typeid) {
		double dlabel; /* store label in case typeid is double*/
		float flabel; /* store label in case typeid is float*/
		int ilabel; /* store label in case typeid is int*/

	case 3:
		if (fread(&ilabel, sizeof(int), 1, docfl) <1) {
			perror("Unable to read label");
			return 0;
		}
		*label = ilabel;
		break;
	case 4:
		if (fread(&flabel, sizeof(float),1,docfl) <1) {
			perror("Unable to read label");
			return 0;
		}
		*label = flabel;
		break;
	case 5:  default:
		if (fread(&dlabel, sizeof(double),1,docfl) <1) {
			perror("Unable to read label");
			return 0;
		}
		*label = dlabel;
		break;
	};
	switch (data_typeid) {
	case 3:
		for (wpos=0; wpos<max_words_doc; ++wpos) 
		{   
			int iweight;
			if (fread(&iweight, sizeof(int), 1, docfl) <1) {
				perror("Unable to read feature vector element");
				return 0;
			}
			(words[wpos]).wnum=wpos+1;
			(words[wpos]).weight=(FVAL)iweight;
		}
		break;
	case 4:
		for (wpos=0; wpos<max_words_doc; ++wpos) 
		{   
			float fweight;
			if (fread(&fweight, sizeof(float), 1,  docfl) <1) {
				perror("Unable to read feature vector element");
				return 0;
			}
			(words[wpos]).wnum=wpos+1;
			(words[wpos]).weight=(FVAL)fweight;
		}
		break;
	case 5:  default:
		for (wpos=0; wpos<max_words_doc; ++wpos) 
		{   
			double dweight;
			if (fread(&dweight, sizeof(double), 1, docfl) <1) {
				perror("Unable to read feature vector element");
				return 0;
			}
			(words[wpos]).wnum=wpos+1;
			(words[wpos]).weight=(FVAL)dweight;
		}
		break;
	};
	(words[wpos]).wnum=0;
	(*numwords)=wpos+1;
	return(1);
}// }}}

void write_binary_model(const char *modelfile, MODEL *model)
{// {{{
	FILE *modelfl;
	long j,i,sv_num;
	SVECTOR *v;

	if(verbosity>=3) {
		printf("Writing model file..."); fflush(stdout);
	}
	if ((modelfl = fopen (modelfile, "wb")) == NULL)
	{ perror (modelfile); exit (1); 
	}
	
	fwrite(&(model->kernel_parm.kernel_type),sizeof(long),1,modelfl);
	fwrite(&(model->kernel_parm.poly_degree),sizeof(long),1,modelfl);
	fwrite(&(model->kernel_parm.rbf_gamma),sizeof(double),1,modelfl);
	fwrite(&(model->kernel_parm.coef_lin),sizeof(double),1,modelfl); 
	fwrite(&(model->kernel_parm.coef_const),sizeof(double),1,modelfl);
	fwrite(&(model->totwords),sizeof(long),1,modelfl);
	
	sv_num=1;
	for(i=1;i<model->sv_num;++i) {
		for(v=model->supvec[i]->fvec;v;v=v->next) 
			sv_num++;
	}
	fwrite(&sv_num, sizeof(long),1,modelfl);
	fwrite(&(model->b), sizeof(double),1,modelfl);

	if(model->kernel_parm.kernel_type == 0) { /* linear kernel */
		add_weight_vector_to_linear_model(model);

		/* save linear wts */
		fwrite(model->lin_weights, sizeof(double),model->totwords+1,modelfl);
	} else {
		for(i=1;i<model->sv_num;++i) {
			for(v=model->supvec[i]->fvec;v;v=v->next) {
				double wt = model->alpha[i]*v->factor;
				fwrite(&wt, sizeof(double),1,modelfl);
				for (j=0; j < v->n_words; ++j) {
					fwrite(&(v->words[j]), sizeof(double),1,modelfl);
				}
			}
		}
	}
	fclose(modelfl);
	if(verbosity>=3) {
		printf("done\n");
	}
}// }}}
