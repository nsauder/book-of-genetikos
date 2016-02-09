#Note: To obtain data input, you'll probably get microarray files from GEO. They come in a '.cel' format and are
#nasty to deal with. Use R to unpack them and pbroduce a text file with all of the intensities
#(you can run the command once from the same directory the .gzip files are in, and produce one text file with all the data)
#To do this, follow instructions here: http://homer.salk.edu/homer/basicTutorial/affymetrix.html
#Note that this method useses RMA (robust multi-array average) for normalization of the raw data

#On a bigger picture, the point of this program is to take an n-dimensional input in the form of a promoter sequence 
#(where n/4 is the number of nucleotides in the promoter) - note for future is there any way to compress this input? maybe do 
#composition analysis - and return a suitable expression profile (i.e. which cell types the promoter is most likely to be on in)

#You should be in the same directory as your gzipped (or not) .CEL files, the annotation file for whatever you want to work with
#(this code is written to intrepret Affymetrix STGene 1.0 files and may require some tweaking for other microarray formats)

#Make a dictionary out of the annotation file
"""Make a dictinoary out of the annotation file with probe number:gene name"""

#Make a promoter database out of 

#Annotate with gene names
"""Make an array out of file containing tab delimited probe number and expression data for different conditions"""
"""Use the annotation dictionary to associate each probe number with a gene name"""


#mouse promoter database sequence (dictionary?)
#Return a list of tuples with promoter sequence in the first tuple, and intensity profile (expression in different files)
#on the other axis

#Run neural net on this input

#Relevant files: 
#Annotation for Immgen files: "1_STGene_Mouse_Affy_Annotations.csv"
#File created with R with all the microarray data to import: "new_data.txt"
import csv
import numpy as np
import pandas as pd

def import_csv(filename):
	csvfile = open(filename,'r')
	csvfile_open = csv.reader(csvfile)
	return csvfile_open

hi = import_csv("1_STGene_Mouse_Affy_Annotations.csv")

class Microarray:
	def __init__(self,annotations,data):
		self.annotations = import_csv(annotations)
		self.data = import_csv(data)


#hi = Microarray("1_STGene_Mouse_Affy_Annotations.csv","new_microarray2.csv")



#Write two different promoter pullers - one taking the 2K nucleotides from the start coordinate of the gene, the other using the gene 
#ID to take the 500 bp from the start of that 


def clean_promoters(filename,promoter_size=500):
	#This returns a list of tuples with gene ame 
	#Opens promoter file in universal newline mode 
	promoters = open(filename,'rU')
	promoters_open = csv.reader(promoters)
	#Extracts # of nucleotides from promoter file - to do
	num_lines = np.ceil(promoter_size/60.0)
	file_lines = []
	for i in promoters_open:
		file_lines.append(i)
	sequences = []
	lines = len(file_lines)
	for i in range(0,lines,int(num_lines+1)):
		a = ""
		for j in range(1,int(num_lines+1)):
			a+=file_lines[i+j][0]
		#This gets the gene name
		gene_name = file_lines[i][0][10:].split(' ',1)[0].split('_',1)[0]
		sequences.append((gene_name,a))
	return sequences

def clean_microarray(data_filename,annotation_filename="1_STGene_Mouse_Affy_Annotations.csv"):
	#Note this is kind of hacky have to label first column as 'probe'
	genes_to_intensities = {}
	data = csv.DictReader(open(data_filename,'rU'))
	annotation = csv.DictReader(open(annotation_filename,'rU'))
	annotation_dict = {}
	failed_matches = []
	for i in annotation:
		annotation_dict[str(i["Probeset ID"])]=str(i["Gene Symbol"])
	for i in data: 
		probe = i['probe']
		#hacky
		if int(probe) > 10344613:
			try:
				if annotation_dict[str(probe)] != '':
					
					genes_to_intensities[str(annotation_dict[str(probe)])]=i
			except:
				failed_matches.append(probe)

	#data = np.genfromtxt("new_microarray2.csv",delimiter=',')
	#data = pd.read_csv(data_filename,sep=',',header=None)
	#data_ndarray = data.as_matrix
	#annotation = pd.read_csv(annotation_filename,sep=',',header=None)
	return genes_to_intensities
	#import microarrray data
	#import probe gene data



hello = clean_promoters("Mouse_promoter_neg_500_to_0.csv")

hi = clean_microarray("new_microarray2.csv")

print hi

#With genes to intensities, map promoters to intensities
#Run the result through a conv net over the promoters to predict the intensity distribution?l

	a = 0
	for row in promoter_df.iterrows():
		if row[1][0][0] == '>':
			a+=1
			if a == 2:
				line_length = row[0]-1
				break


		if a == 0:
			gene_name = row[1][0].split(' ',2)[1]
		elif row[1][0][0] == '>':
			genes_to_promoters[gene_name] = new_seq
			gene_name = row[1][0].split(' ',2)[1]
			new_seq = ''
		elif a == 30:
			break
		else:
			new_seq = new_seq.join(row[1][0])
			a+=1




import pandas as pd
df = pd.read_csv("1_STGene_Mouse_Affy_Annotations.csv")
df.head()
head 1_STGene_Mouse_Affy_Annotations.csv
!head 1_STGene_Mouse_Affy_Annotations.csv
!head -n 3 1_STGene_Mouse_Affy_Annotations.csv
df.shape
df.head()
df2 = pd.read_csv("new_microarray2.csv")
df2.head()
pd.isnull(df["Gene Symbol"]).mean()
df.columns
ann_df = ann_df[~pd.isnull(ann_df["Gene Symbol"])]
ann_df.shape
data_df = df2
row = data_df.iterrows().next()[1]
row.to_dict()
%paste
probeset_to_gene_symbol.iteritems().next()
gene_symbol_to_row.iteritems().next()
%paste
gene_symbol_to_row.iteritems().next()
len(gene_symbol_to_row)
max(list(gene_symbol_to_row.iteritems()), key=lambda x: len(x[1]))
foo = max(list(gene_symbol_to_row.iteritems()), key=lambda x: len(x[1]))

: len(foo[1])
4]: foo[0]
