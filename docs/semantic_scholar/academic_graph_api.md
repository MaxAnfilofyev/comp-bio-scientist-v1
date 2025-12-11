Academic Graph API (1.0)
Download OpenAPI specification:Download

Fetch paper and author data from the Semantic Scholar Academic Graph (S2AG).

Some things to note:

If you are using an API key, it must be set in the header x-api-key (case-sensitive).
We have two different IDs for a single paper:
paperId - string - The primary way to identify papers when using our website or this API
corpusId - int64 - A second way to identify papers. Our datasets use corpusId when pointing to papers.
Other useful resources
Overview
allenai/s2-folks
FAQ in allenai/s2folks
Paper Data
Suggest paper query completions
To support interactive query-completion, return minimal information about papers matching a partial query

Example: https://api.semanticscholar.org/graph/v1/paper/autocomplete?query=semanti

query Parameters
query
required
string
Plain-text partial query string. Will be truncated to first 100 characters.

Responses
200 Batch of papers with default or requested fields
400 Bad query parameters

get
/paper/autocomplete
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"matches": [
{}
]
}
Get details for multiple papers at once
Fields is a single-value string parameter, not a multi-value one.
It is a query parameter, not to be submitted in the POST request's body.
In python:

r = requests.post(
    'https://api.semanticscholar.org/graph/v1/paper/batch',
    params={'fields': 'referenceCount,citationCount,title'},
    json={"ids": ["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}
)
print(json.dumps(r.json(), indent=2))

[
  {
    "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
    "title": "Construction of the Literature Graph in Semantic Scholar",
    "referenceCount": 27,
    "citationCount": 299
  },
  {
    "paperId": "f712fab0d58ae6492e3cdfc1933dae103ec12d5d",
    "title": "Reinfection and low cross-immunity as drivers of epidemic resurgence under high seroprevalence: a model-based approach with application to Amazonas, Brazil",
    "referenceCount": 13,
    "citationCount": 0
  }
]
Other Examples:

https://api.semanticscholar.org/graph/v1/paper/batch
{"ids":["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}
Returns details for 2 papers.
Each paper has its paperId and title.
https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,isOpenAccess,openAccessPdf,authors
{"ids":["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}
Returns all requested info plus paper IDs for 2 papers.

Limitations:
Can only process 500 paper ids at a time.
Can only return up to 10 MB of data at a time.
Can only return up to 9999 citations at a time.
For a list of supported IDs reference the "Details about a paper" endpoint.
query Parameters
fields	
string
A comma-separated list of the fields to be returned. See the contents of Response Schema below for a list of all available fields that can be returned. The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Use a period (“.”) for fields that have version numbers or subfields, such as the embedding, authors, citations, and references fields:

When requesting authors, the authorId and name subfields are returned by default. To request other subfields, use the format author.url,author.paperCount, etc. See the Response Schema below for available subfields.
When requesting citations and references, the paperId and title subfields are returned by default. To request other subfields, use the format citations.title,citations.abstract, etc. See the Response Schema below for available subfields.
When requesting embedding, the default Spector embedding version is v1. Specify embedding.specter_v2 to select v2 embeddings.
Examples:
fields=title,url
fields=title,embedding.specter_v2
fields=title,authors,citations.title,citations.abstract
Request Body schema: application/json
required
ids	
Array of strings
Responses
200 List of papers with default or requested fields
400 Bad query parameters

post
/paper/batch
Request samples
Payload
Content type
application/json

Copy
Expand allCollapse all
{
"ids": [
"649def34f8be52c8b66281af98ae884c09aef38b"
]
}
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"paperId": "5c5751d45e298cea054f32b392c12c61027d2fe7",
"corpusId": 215416146,
"externalIds": {
"MAG": "3015453090",
"DBLP": "conf/acl/LoWNKW20",
"ACL": "2020.acl-main.447",
"DOI": "10.18653/V1/2020.ACL-MAIN.447",
"CorpusId": 215416146
},
"url": "https://www.semanticscholar.org/paper/5c5751d45e298cea054f32b392c12c61027d2fe7",
"title": "Construction of the Literature Graph in Semantic Scholar",
"abstract": "We describe a deployed scalable system for organizing published scientific literature into a heterogeneous graph to facilitate algorithmic manipulation and discovery.",
"venue": "Annual Meeting of the Association for Computational Linguistics",
"publicationVenue": {
"id": "1e33b3be-b2ab-46e9-96e8-d4eb4bad6e44",
"name": "Annual Meeting of the Association for Computational Linguistics",
"type": "conference",
"alternate_names": [],
"url": "https://www.aclweb.org/anthology/venues/acl/"
},
"year": 1997,
"referenceCount": 59,
"citationCount": 453,
"influentialCitationCount": 90,
"isOpenAccess": true,
"openAccessPdf": {
"url": "https://www.aclweb.org/anthology/2020.acl-main.447.pdf",
"status": "HYBRID",
"license": "CCBY",
"disclaimer": "Notice: This snippet is extracted from the open access paper or abstract available at https://aclanthology.org/2020.acl-main.447, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use."
},
"fieldsOfStudy": [
"Computer Science"
],
"s2FieldsOfStudy": [
{},
{},
{}
],
"publicationTypes": [
"Journal Article",
"Review"
],
"publicationDate": "2024-04-29",
"journal": {
"volume": "40",
"pages": "116 - 135",
"name": "IETE Technical Review"
},
"citationStyles": {
"bibtex": "@['JournalArticle', 'Conference']{Ammar2018ConstructionOT,\n author = {Waleed Ammar and Dirk Groeneveld and Chandra Bhagavatula and Iz Beltagy and Miles Crawford and Doug Downey and Jason Dunkelberger and Ahmed Elgohary and Sergey Feldman and Vu A. Ha and Rodney Michael Kinney and Sebastian Kohlmeier and Kyle Lo and Tyler C. Murray and Hsu-Han Ooi and Matthew E. Peters and Joanna L. Power and Sam Skjonsberg and Lucy Lu Wang and Christopher Wilhelm and Zheng Yuan and Madeleine van Zuylen and Oren Etzioni},\n booktitle = {NAACL},\n pages = {84-91},\n title = {Construction of the Literature Graph in Semantic Scholar},\n year = {2018}\n}\n"
},
"authors": [
{}
],
"citations": [
{}
],
"references": [
{}
],
"embedding": {
"model": "specter@v0.1.1",
"vector": []
},
"tldr": {
"model": "tldr@v2.0.0",
"text": "This paper reduces literature graph construction into familiar NLP tasks, point out research challenges due to differences from standard formulations of these tasks, and report empirical results for each task."
},
"textAvailability": "string"
}
Paper relevance search
Examples:

https://api.semanticscholar.org/graph/v1/paper/search?query=covid+vaccination&offset=100&limit=3
Returns with total=576278, offset=100, next=103, and data is a list of 3 papers.
Each paper has its paperId and title.
https://api.semanticscholar.org/graph/v1/paper/search?query=covid&fields=url,abstract,authors
Returns with total=639637, offset=0, next=100, and data is a list of 100 papers.
Each paper has paperId, url, abstract, and a list of authors.
Each author under that list has authorId and name.
https://api.semanticscholar.org/graph/v1/paper/search?query=totalGarbageNonsense
Returns with total=0, offset=0, and data is a list of 0 papers.
https://api.semanticscholar.org/graph/v1/paper/search?query=covid&year=2020-2023&openAccessPdf&fieldsOfStudy=Physics,Philosophy&fields=title,year,authors
Returns with total=8471, offset=0, next=10, and data is a list of 10 papers.
Filters to include only papers published between 2020-2023.
Filters to include only papers with open access PDFs.
Filters to include only papers that have a field of study either matching Physics or Philosophy.
Each paper has the fields paperId, title, year, and authors.

Limitations:
Can only return up to 1,000 relevance-ranked results. For larger queries, see "/search/bulk" or the Datasets API.
Can only return up to 10 MB of data at a time.
query Parameters
query
required
string
A plain-text search query string.

No special query syntax is supported.
Hyphenated query terms yield no matches (replace it with space to find matches)
See our blog post for a description of our search relevance algorithm.

Example: graph/v1/paper/search?query=generative ai

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Use a period (“.”) for fields that have version numbers or subfields, such as the embedding, authors, citations, and references fields:

When requesting authors, the authorId and name subfields are returned by default. To request other subfields, use the format author.url,author.paperCount, etc. See the Response Schema below for available subfields.
When requesting citations and references, the paperId and title subfields are returned by default. To request other subfields, use the format citations.title,citations.abstract, etc. See the Response Schema below for available subfields.
When requesting embedding, the default Spector embedding version is v1. Specify embedding.specter_v2 to select v2 embeddings.
Examples:
fields=title,url
fields=title,embedding.specter_v2
fields=title,authors,citations.title,citations.abstract
publicationTypes	
string
Restricts results to any of the following paper publication types:

Review
JournalArticle
CaseReport
ClinicalTrial
Conference
Dataset
Editorial
LettersAndComments
MetaAnalysis
News
Study
Book
BookSection
Use a comma-separated list to include papers with any of the listed publication types.

Example: Review,JournalArticle will return papers with publication types Review and/or JournalArticle.

openAccessPdf	
string
Restricts results to only include papers with a public PDF. This parameter does not accept any values.

minCitationCount	
string
Restricts results to only include papers with the minimum number of citations.

Example: minCitationCount=200

publicationDateOrYear	
string
Restricts results to the given range of publication dates or years (inclusive). Accepts the format <startDate>:<endDate> with each date in YYYY-MM-DD format.

Each term is optional, allowing for specific dates, fixed ranges, or open-ended ranges. In addition, prefixes are supported as a shorthand, e.g. 2020-06 matches all dates in June 2020.

Specific dates are not known for all papers, so some records returned with this filter will have a null value for publicationDate. year, however, will always be present. For records where a specific publication date is not known, they will be treated as if published on January 1st of their publication year.

Examples:

2019-03-05 on March 5th, 2019
2019-03 during March 2019
2019 during 2019
2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
1981-08-25: on or after August 25th, 1981
:2015-01 before or on January 31st, 2015
2015:2020 between January 1st, 2015 and December 31st, 2020
year	
string
Restricts results to the given publication year or range of years (inclusive).

Examples:

2019 in 2019
2016-2020 as early as 2016 or as late as 2020
2010- during or after 2010
-2015 before or during 2015
venue	
string
Restricts results to papers published in the given venues, formatted as a comma-separated list.

Input could also be an ISO4 abbreviation. Examples include:

Nature
New England Journal of Medicine
Radiology
N. Engl. J. Med.
Example: Nature,Radiology will return papers from venues Nature and/or Radiology.

fieldsOfStudy	
string
Restricts results to papers in the given fields of study, formatted as a comma-separated list:

Computer Science
Medicine
Chemistry
Biology
Materials Science
Physics
Geology
Psychology
Art
History
Geography
Sociology
Business
Political Science
Economics
Philosophy
Mathematics
Engineering
Environmental Science
Agricultural and Food Sciences
Education
Law
Linguistics
Example: Physics,Mathematics will return papers with either Physics or Mathematics in their list of fields-of-study.

offset	
integer
Default: 0
Used for pagination. When returning a list of results, start with the element at this position in the list.

limit	
integer
Default: 100
The maximum number of results to return.
Must be <= 100

Responses
200 Batch of papers with default or requested fields
400 Bad query parameters

get
/paper/search
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"total": 15117,
"offset": 0,
"next": 0,
"data": [
{}
]
}
Paper bulk search
Behaves similarly to /paper/search, but is intended for bulk retrieval of basic paper data without search relevance:

Text query is optional and supports boolean logic for document matching.
Papers can be filtered using various criteria.
Up to 1,000 papers will be returned in each call.
If there are more matching papers, a continuation "token" will be present.
The query can be repeated with the token param added to efficiently continue fetching matching papers.

Returns a structure with an estimated total matches, batch of matching papers, and a continuation token if more results are available.
Limitations:
Nested paper data, such as citations, references, etc, is not available via this method.
Up to 10,000,000 papers can be fetched via this method. For larger needs, please use the Datasets API to retrieve full copies of the corpus.
query Parameters
query
required
string
Text query that will be matched against the paper's title and abstract. All terms are stemmed in English. By default all terms in the query must be present in the paper.

The match query supports the following syntax:

+ for AND operation
| for OR operation
- negates a term
" collects terms into a phrase
* can be used to match a prefix
( and ) for precedence
~N after a word matches within the edit distance of N (Defaults to 2 if N is omitted)
~N after a phrase matches with the phrase terms separated up to N terms apart (Defaults to 2 if N is omitted)
Examples:

fish ladder matches papers that contain "fish" and "ladder"
fish -ladder matches papers that contain "fish" but not "ladder"
fish | ladder matches papers that contain "fish" or "ladder"
"fish ladder" matches papers that contain the phrase "fish ladder"
(fish ladder) | outflow matches papers that contain "fish" and "ladder" OR "outflow"
fish~ matches papers that contain "fish", "fist", "fihs", etc.
"fish ladder"~3 mathces papers that contain the phrase "fish ladder" or "fish is on a ladder"
token	
string
Used for pagination. This string token is provided when the original query returns, and is used to fetch the next batch of papers. Each call will return a new token.

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned.

The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Examples: https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=covid&fields=venue,s2FieldsOfStudy

sort	
string
Provides the option to sort the results by the following fields:

paperId
publicationDate
citationCount
Uses the format field:order. Ties are broken by paperId. The default field is paperId and the default order is asc. Records for which the sort value are not defined will appear at the end of sort, regardless of asc/desc order.

Examples:
publicationDate:asc - return oldest papers first.
citationCount:desc - return most highly-cited papers first.
paperId - return papers in ID order, low-to-high.

Please be aware that if the relevant data changes while paging through results, records can be returned in an unexpected way. The default paperId sort avoids this edge case.
publicationTypes	
string
Restricts results to any of the following paper publication types:

Review
JournalArticle
CaseReport
ClinicalTrial
Conference
Dataset
Editorial
LettersAndComments
MetaAnalysis
News
Study
Book
BookSection
Use a comma-separated list to include papers with any of the listed publication types.

Example: Review,JournalArticle will return papers with publication types Review and/or JournalArticle.

openAccessPdf	
string
Restricts results to only include papers with a public PDF. This parameter does not accept any values.

minCitationCount	
string
Restricts results to only include papers with the minimum number of citations.

Example: minCitationCount=200

publicationDateOrYear	
string
Restricts results to the given range of publication dates or years (inclusive). Accepts the format <startDate>:<endDate> with each date in YYYY-MM-DD format.

Each term is optional, allowing for specific dates, fixed ranges, or open-ended ranges. In addition, prefixes are supported as a shorthand, e.g. 2020-06 matches all dates in June 2020.

Specific dates are not known for all papers, so some records returned with this filter will have a null value for publicationDate. year, however, will always be present. For records where a specific publication date is not known, they will be treated as if published on January 1st of their publication year.

Examples:

2019-03-05 on March 5th, 2019
2019-03 during March 2019
2019 during 2019
2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
1981-08-25: on or after August 25th, 1981
:2015-01 before or on January 31st, 2015
2015:2020 between January 1st, 2015 and December 31st, 2020
year	
string
Restricts results to the given publication year or range of years (inclusive).

Examples:

2019 in 2019
2016-2020 as early as 2016 or as late as 2020
2010- during or after 2010
-2015 before or during 2015
venue	
string
Restricts results to papers published in the given venues, formatted as a comma-separated list.

Input could also be an ISO4 abbreviation. Examples include:

Nature
New England Journal of Medicine
Radiology
N. Engl. J. Med.
Example: Nature,Radiology will return papers from venues Nature and/or Radiology.

fieldsOfStudy	
string
Restricts results to papers in the given fields of study, formatted as a comma-separated list:

Computer Science
Medicine
Chemistry
Biology
Materials Science
Physics
Geology
Psychology
Art
History
Geography
Sociology
Business
Political Science
Economics
Philosophy
Mathematics
Engineering
Environmental Science
Agricultural and Food Sciences
Education
Law
Linguistics
Example: Physics,Mathematics will return papers with either Physics or Mathematics in their list of fields-of-study.

Responses
200 Batch of papers with default or requested fields
400 Bad query parameters

get
/paper/search/bulk
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"total": 15117,
"token": "SDKJFHSDKFHWIEFSFSGHEIURYC",
"data": [
{}
]
}
Paper title search
Behaves similarly to /paper/search, but is intended for retrieval of a single paper based on closest title match to given query. Examples:

https://api.semanticscholar.org/graph/v1/paper/search/match?query=Construction of the Literature Graph in Semantic Scholar
Returns a single paper that is the closest title match.
Each paper has its paperId, title, and matchScore as well as any other requested fields.
https://api.semanticscholar.org/graph/v1/paper/search/match?query=totalGarbageNonsense
Returns with a 404 error and a "Title match not found" message.

Limitations:
Will only return the single highest match result.
query Parameters
query
required
string
A plain-text search query string.

No special query syntax is supported.
See our blog post for a description of our search relevance algorithm.

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Use a period (“.”) for fields that have version numbers or subfields, such as the embedding, authors, citations, and references fields:

When requesting authors, the authorId and name subfields are returned by default. To request other subfields, use the format author.url,author.paperCount, etc. See the Response Schema below for available subfields.
When requesting citations and references, the paperId and title subfields are returned by default. To request other subfields, use the format citations.title,citations.abstract, etc. See the Response Schema below for available subfields.
When requesting embedding, the default Spector embedding version is v1. Specify embedding.specter_v2 to select v2 embeddings.
Examples:
fields=title,url
fields=title,embedding.specter_v2
fields=title,authors,citations.title,citations.abstract
publicationTypes	
string
Restricts results to any of the following paper publication types:

Review
JournalArticle
CaseReport
ClinicalTrial
Conference
Dataset
Editorial
LettersAndComments
MetaAnalysis
News
Study
Book
BookSection
Use a comma-separated list to include papers with any of the listed publication types.

Example: Review,JournalArticle will return papers with publication types Review and/or JournalArticle.

openAccessPdf	
string
Restricts results to only include papers with a public PDF. This parameter does not accept any values.

minCitationCount	
string
Restricts results to only include papers with the minimum number of citations.

Example: minCitationCount=200

publicationDateOrYear	
string
Restricts results to the given range of publication dates or years (inclusive). Accepts the format <startDate>:<endDate> with each date in YYYY-MM-DD format.

Each term is optional, allowing for specific dates, fixed ranges, or open-ended ranges. In addition, prefixes are supported as a shorthand, e.g. 2020-06 matches all dates in June 2020.

Specific dates are not known for all papers, so some records returned with this filter will have a null value for publicationDate. year, however, will always be present. For records where a specific publication date is not known, they will be treated as if published on January 1st of their publication year.

Examples:

2019-03-05 on March 5th, 2019
2019-03 during March 2019
2019 during 2019
2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
1981-08-25: on or after August 25th, 1981
:2015-01 before or on January 31st, 2015
2015:2020 between January 1st, 2015 and December 31st, 2020
year	
string
Restricts results to the given publication year or range of years (inclusive).

Examples:

2019 in 2019
2016-2020 as early as 2016 or as late as 2020
2010- during or after 2010
-2015 before or during 2015
venue	
string
Restricts results to papers published in the given venues, formatted as a comma-separated list.

Input could also be an ISO4 abbreviation. Examples include:

Nature
New England Journal of Medicine
Radiology
N. Engl. J. Med.
Example: Nature,Radiology will return papers from venues Nature and/or Radiology.

fieldsOfStudy	
string
Restricts results to papers in the given fields of study, formatted as a comma-separated list:

Computer Science
Medicine
Chemistry
Biology
Materials Science
Physics
Geology
Psychology
Art
History
Geography
Sociology
Business
Political Science
Economics
Philosophy
Mathematics
Engineering
Environmental Science
Agricultural and Food Sciences
Education
Law
Linguistics
Example: Physics,Mathematics will return papers with either Physics or Mathematics in their list of fields-of-study.

Responses
200 Best Title match paper with default or requested fields
400 Bad query parameters
404 No title match

get
/paper/search/match
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"data": [
{}
]
}
Details about a paper
Examples:

https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b
Returns a paper with its paperId and title.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b?fields=url,year,authors
Returns the paper's paperId, url, year, and list of authors.
Each author has authorId and name.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b?fields=citations.authors
Returns the paper's paperId and list of citations.
Each citation has its paperId plus its list of authors.
Each author has their 2 always included fields of authorId and name.

Limitations:
Can only return up to 10 MB of data at a time.
path Parameters
paper_id
required
string
The following types of IDs are supported:

<sha> - a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
CorpusId:<id> - a Semantic Scholar numerical ID, e.g. CorpusId:215416146
DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
PMID:<id> - PubMed/Medline, e.g. PMID:19872477
PMCID:<id> - PubMed Central, e.g. PMCID:2323736
URL:<url> - URL from one of the sites listed below, e.g. URL:https://arxiv.org/abs/2106.15928v1
URLs are recognized from the following sites:

semanticscholar.org
arxiv.org
aclweb.org
acm.org
biorxiv.org
query Parameters
fields	
string
A comma-separated list of the fields to be returned. See the contents of Response Schema below for a list of all available fields that can be returned. The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Use a period (“.”) for fields that have version numbers or subfields, such as the embedding, authors, citations, and references fields:

When requesting authors, the authorId and name subfields are returned by default. To request other subfields, use the format author.url,author.paperCount, etc. See the Response Schema below for available subfields.
When requesting citations and references, the paperId and title subfields are returned by default. To request other subfields, use the format citations.title,citations.abstract, etc. See the Response Schema below for available subfields.
When requesting embedding, the default Spector embedding version is v1. Specify embedding.specter_v2 to select v2 embeddings.
Examples:
fields=title,url
fields=title,embedding.specter_v2
fields=title,authors,citations.title,citations.abstract
Responses
200 Paper with default or requested fields
400 Bad query parameters
404 Bad paper id

get
/paper/{paper_id}
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"paperId": "5c5751d45e298cea054f32b392c12c61027d2fe7",
"corpusId": 215416146,
"externalIds": {
"MAG": "3015453090",
"DBLP": "conf/acl/LoWNKW20",
"ACL": "2020.acl-main.447",
"DOI": "10.18653/V1/2020.ACL-MAIN.447",
"CorpusId": 215416146
},
"url": "https://www.semanticscholar.org/paper/5c5751d45e298cea054f32b392c12c61027d2fe7",
"title": "Construction of the Literature Graph in Semantic Scholar",
"abstract": "We describe a deployed scalable system for organizing published scientific literature into a heterogeneous graph to facilitate algorithmic manipulation and discovery.",
"venue": "Annual Meeting of the Association for Computational Linguistics",
"publicationVenue": {
"id": "1e33b3be-b2ab-46e9-96e8-d4eb4bad6e44",
"name": "Annual Meeting of the Association for Computational Linguistics",
"type": "conference",
"alternate_names": [],
"url": "https://www.aclweb.org/anthology/venues/acl/"
},
"year": 1997,
"referenceCount": 59,
"citationCount": 453,
"influentialCitationCount": 90,
"isOpenAccess": true,
"openAccessPdf": {
"url": "https://www.aclweb.org/anthology/2020.acl-main.447.pdf",
"status": "HYBRID",
"license": "CCBY",
"disclaimer": "Notice: This snippet is extracted from the open access paper or abstract available at https://aclanthology.org/2020.acl-main.447, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use."
},
"fieldsOfStudy": [
"Computer Science"
],
"s2FieldsOfStudy": [
{},
{},
{}
],
"publicationTypes": [
"Journal Article",
"Review"
],
"publicationDate": "2024-04-29",
"journal": {
"volume": "40",
"pages": "116 - 135",
"name": "IETE Technical Review"
},
"citationStyles": {
"bibtex": "@['JournalArticle', 'Conference']{Ammar2018ConstructionOT,\n author = {Waleed Ammar and Dirk Groeneveld and Chandra Bhagavatula and Iz Beltagy and Miles Crawford and Doug Downey and Jason Dunkelberger and Ahmed Elgohary and Sergey Feldman and Vu A. Ha and Rodney Michael Kinney and Sebastian Kohlmeier and Kyle Lo and Tyler C. Murray and Hsu-Han Ooi and Matthew E. Peters and Joanna L. Power and Sam Skjonsberg and Lucy Lu Wang and Christopher Wilhelm and Zheng Yuan and Madeleine van Zuylen and Oren Etzioni},\n booktitle = {NAACL},\n pages = {84-91},\n title = {Construction of the Literature Graph in Semantic Scholar},\n year = {2018}\n}\n"
},
"authors": [
{}
],
"citations": [
{}
],
"references": [
{}
],
"embedding": {
"model": "specter@v0.1.1",
"vector": []
},
"tldr": {
"model": "tldr@v2.0.0",
"text": "This paper reduces literature graph construction into familiar NLP tasks, point out research challenges due to differences from standard formulations of these tasks, and report empirical results for each task."
},
"textAvailability": "string"
}
Details about a paper's authors
Examples:

https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/authors
Returns with offset=0, and data is a list of all 3 authors.
Each author has their authorId and name
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/authors?fields=affiliations,papers&limit=2
Returns with offset=0, next=2, and data is a list of 2 authors.
Each author has their authorId, affiliations, and list of papers.
Each paper has its paperId and title.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/authors?fields=url,papers.year,papers.authors&offset=2
Returns with offset=2, and data is a list containing the last author.
This author has their authorId, url, and list of papers.
Each paper has its paperId, year, and list of authors.
In that list of authors, each author has their authorId and name.
path Parameters
paper_id
required
string
The following types of IDs are supported:

<sha> - a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
CorpusId:<id> - a Semantic Scholar numerical ID, e.g. CorpusId:215416146
DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
PMID:<id> - PubMed/Medline, e.g. PMID:19872477
PMCID:<id> - PubMed Central, e.g. PMCID:2323736
URL:<url> - URL from one of the sites listed below, e.g. URL:https://arxiv.org/abs/2106.15928v1
URLs are recognized from the following sites:

semanticscholar.org
arxiv.org
aclweb.org
acm.org
biorxiv.org
query Parameters
offset	
integer
Default: 0
Used for pagination. When returning a list of results, start with the element at this position in the list.

limit	
integer
Default: 100
The maximum number of results to return.
Must be <= 1000

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. The authorId field is always returned. If the fields parameter is omitted, only the authorId and name will be returned.

Use a period (“.”) for subfields of papers.

Examples:

fields=name,affiliations,papers
fields=url,papers.year,papers.authors
Responses
200 List of Authors with default or requested fields
400 Bad query parameters
404 Bad paper id

get
/paper/{paper_id}/authors
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"offset": 0,
"next": 0,
"data": [
{}
]
}
Details about a paper's citations
Fetch details about the papers that cite this paper (i.e. papers in whose bibliography this paper appears)

Examples:

Let's suppose that the paper in the examples below has 1600 citations...
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/citations
Returns with offset=0, next=100, and data is a list of 100 citations.
Each citation has a citingPaper which contains its paperId and title.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/citations?fields=contexts,intents,isInfluential,abstract&offset=200&limit=10
Returns with offset=200, next=210, and data is a list of 10 citations.
Each citation has contexts, intents, isInfluential, and a citingPaper which contains its paperId and abstract.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/citations?fields=authors&offset=1500&limit=500
Returns with offset=1500, and data is a list of the last 100 citations.
Each citation has a citingPaper which contains its paperId plus a list of authors
The authors under each citingPaper has their authorId and name.
path Parameters
paper_id
required
string
The following types of IDs are supported:

<sha> - a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
CorpusId:<id> - a Semantic Scholar numerical ID, e.g. CorpusId:215416146
DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
PMID:<id> - PubMed/Medline, e.g. PMID:19872477
PMCID:<id> - PubMed Central, e.g. PMCID:2323736
URL:<url> - URL from one of the sites listed below, e.g. URL:https://arxiv.org/abs/2106.15928v1
URLs are recognized from the following sites:

semanticscholar.org
arxiv.org
aclweb.org
acm.org
biorxiv.org
query Parameters
publicationDateOrYear	
string
Restricts results to the given range of publication dates or years (inclusive). Accepts the format <startDate>:<endDate> with each date in YYYY-MM-DD format.

Each term is optional, allowing for specific dates, fixed ranges, or open-ended ranges. In addition, prefixes are supported as a shorthand, e.g. 2020-06 matches all dates in June 2020.

Specific dates are not known for all papers, so some records returned with this filter will have a null value for publicationDate. year, however, will always be present. For records where a specific publication date is not known, they will be treated as if published on January 1st of their publication year.

Examples:

2019-03-05 on March 5th, 2019
2019-03 during March 2019
2019 during 2019
2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
1981-08-25: on or after August 25th, 1981
:2015-01 before or on January 31st, 2015
2015:2020 between January 1st, 2015 and December 31st, 2020
offset	
integer
Default: 0
Used for pagination. When returning a list of results, start with the element at this position in the list.

limit	
integer
Default: 100
The maximum number of results to return.
Must be <= 1000

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. If the fields parameter is omitted, only the paperId and title will be returned.

Request fields nested within citedPaper the same way as fields like contexts.

Examples:

fields=contexts,isInfluential
fields=contexts,title,authors
Responses
200 Batch of citations with default or requested fields
400 Bad query parameters
404 Bad paper id

get
/paper/{paper_id}/citations
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"offset": 0,
"next": 0,
"data": [
{}
]
}
Details about a paper's references
Fetch details about the papers cited by this paper (i.e. appearing in this paper's bibliography)

Examples:

Let's suppose that the paper in the examples below has 1600 references...
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/references
Returns with offset=0, next=100, and data is a list of 100 references.
Each reference has a citedPaper which contains its paperId and title.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/references?fields=contexts,intents,isInfluential,abstract&offset=200&limit=10
Returns with offset=200, next=210, and data is a list of 10 references.
Each reference has contexts, intents, isInfluential, and a citedPaper which contains its paperId and abstract.
https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/references?fields=authors&offset=1500&limit=500
Returns with offset=1500, and data is a list of the last 100 references.
Each reference has a citedPaper which contains its paperId plus a list of authors
The authors under each citedPaper has their authorId and name.
path Parameters
paper_id
required
string
The following types of IDs are supported:

<sha> - a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
CorpusId:<id> - a Semantic Scholar numerical ID, e.g. CorpusId:215416146
DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
PMID:<id> - PubMed/Medline, e.g. PMID:19872477
PMCID:<id> - PubMed Central, e.g. PMCID:2323736
URL:<url> - URL from one of the sites listed below, e.g. URL:https://arxiv.org/abs/2106.15928v1
URLs are recognized from the following sites:

semanticscholar.org
arxiv.org
aclweb.org
acm.org
biorxiv.org
query Parameters
offset	
integer
Default: 0
Used for pagination. When returning a list of results, start with the element at this position in the list.

limit	
integer
Default: 100
The maximum number of results to return.
Must be <= 1000

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. If the fields parameter is omitted, only the paperId and title will be returned.

Request fields nested within citedPaper the same way as fields like contexts.

Examples:

fields=contexts,isInfluential
fields=contexts,title,authors
Responses
200 Batch of references with default or requested fields
400 Bad query parameters
404 Bad paper id

get
/paper/{paper_id}/references
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"offset": 0,
"next": 0,
"data": [
{}
]
}
Author Data
Get details for multiple authors at once
Fields is a single-value string parameter, not a multi-value one.
It is a query parameter, not to be submitted in the POST request's body.
In python:

r = requests.post(
    'https://api.semanticscholar.org/graph/v1/author/batch',
    params={'fields': 'name,hIndex,citationCount'},
    json={"ids":["1741101", "1780531"]}
)
print(json.dumps(r.json(), indent=2))

[
  {
    "authorId": "1741101",
    "name": "Oren Etzioni",
    "citationCount": 34803,
    "hIndex": 86
  },
  {
    "authorId": "1780531",
    "name": "Daniel S. Weld",
    "citationCount": 35526,
    "hIndex": 89
  }
]
Other Examples:

https://api.semanticscholar.org/graph/v1/author/batch
{"ids":["1741101", "1780531", "48323507"]}
Returns details for 3 authors.
Each author returns the field authorId and name if no other fields are specified.
https://api.semanticscholar.org/graph/v1/author/batch?fields=url,name,paperCount,papers,papers.title,papers.openAccessPdf
{"ids":["1741101", "1780531", "48323507"]}
Returns authorID, url, name, paperCount, and list of papers for 3 authors.
Each paper has its paperID, title, and link if available.

Limitations:
Can only process 1,000 author ids at a time.
Can only return up to 10 MB of data at a time.
query Parameters
fields	
string
A comma-separated list of the fields to be returned. See the contents of Response Schema below for a list of all available fields that can be returned. The authorId field is always returned. If the fields parameter is omitted, only the authorId and name will be returned.

Use a period (“.”) for subfields of papers.

Examples:

fields=name,affiliations,papers
fields=url,papers.year,papers.authors
Request Body schema: application/json
required
ids	
Array of strings
Responses
200 List of authors with default or requested fields
400 Bad query parameters

post
/author/batch
Request samples
Payload
Content type
application/json

Copy
Expand allCollapse all
{
"ids": [
"1741101"
]
}
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"authorId": "1741101",
"externalIds": {
"DBLP": []
},
"url": "https://www.semanticscholar.org/author/1741101",
"name": "Oren Etzioni",
"affiliations": [
"Allen Institute for AI"
],
"homepage": "https://allenai.org/",
"paperCount": 10,
"citationCount": 50,
"hIndex": 5,
"papers": [
{}
]
}
Search for authors by name
Specifying papers fields in the request will return all papers linked to each author in the results. Set a limit on the search results to reduce output size and latency.

Examples:

https://api.semanticscholar.org/graph/v1/author/search?query=adam+smith
Returns with total=490, offset=0, next=100, and data is a list of 100 authors.
Each author has their authorId and name.
https://api.semanticscholar.org/graph/v1/author/search?query=adam+smith&fields=name,url,papers.title,papers.year&limit=5
Returns with total=490, offset=0, next=5, and data is a list of 5 authors.
Each author has authorId, name, url, and a list of their papers title and year.
https://api.semanticscholar.org/graph/v1/author/search?query=totalGarbageNonsense
Returns with total = 0, offset=0, and data is a list of 0 author.

Limitations:
Can only return up to 10 MB of data at a time.
query Parameters
offset	
integer
Default: 0
Used for pagination. When returning a list of results, start with the element at this position in the list.

limit	
integer
Default: 100
The maximum number of results to return.
Must be <= 1000

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. The authorId field is always returned. If the fields parameter is omitted, only the authorId and name will be returned.

Use a period (“.”) for subfields of papers.

Examples:

fields=name,affiliations,papers
fields=url,papers.year,papers.authors
query
required
string
A plain-text search query string.

No special query syntax is supported.
Hyphenated query terms yield no matches (replace it with space to find matches)
Responses
200 Batch of authors with default or requested fields
400 Bad query parameters

get
/author/search
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"total": 15117,
"offset": 0,
"next": 0,
"data": [
{}
]
}
Details about an author
Examples:

https://api.semanticscholar.org/graph/v1/author/1741101
Returns the author's authorId and name.
https://api.semanticscholar.org/graph/v1/author/1741101?fields=url,papers
Returns the author's authorId, url, and list of papers.
Each paper has its paperId plus its title.
https://api.semanticscholar.org/graph/v1/author/1741101?fields=url,papers.abstract,papers.authors
Returns the author's authorId, url, and list of papers.
Each paper has its paperId, abstract, and list of authors.
In that list of authors, each author has their authorId and name.

Limitations:
Can only return up to 10 MB of data at a time.
path Parameters
author_id
required
string
query Parameters
fields	
string
A comma-separated list of the fields to be returned. See the contents of Response Schema below for a list of all available fields that can be returned. The authorId field is always returned. If the fields parameter is omitted, only the authorId and name will be returned.

Use a period (“.”) for subfields of papers.

Examples:

fields=name,affiliations,papers
fields=url,papers.year,papers.authors
Responses
200 Author with default or requested fields
400 Bad query parameters
404 Bad paper id

get
/author/{author_id}
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"authorId": "1741101",
"externalIds": {
"DBLP": []
},
"url": "https://www.semanticscholar.org/author/1741101",
"name": "Oren Etzioni",
"affiliations": [
"Allen Institute for AI"
],
"homepage": "https://allenai.org/",
"paperCount": 10,
"citationCount": 50,
"hIndex": 5,
"papers": [
{}
]
}
Details about an author's papers
Fetch the papers of an author in batches.
Only retrieves the most recent 10,000 citations/references for papers belonging to the batch.
To retrieve the full set of citations for a paper, use the /paper/{paper_id}/citations endpoint

Examples:

https://api.semanticscholar.org/graph/v1/author/1741101/papers
Return with offset=0, and data is a list of the first 100 papers.
Each paper has its paperId and title.
https://api.semanticscholar.org/graph/v1/author/1741101/papers?fields=url,year,authors&limit=2
Returns with offset=0, next=2, and data is a list of 2 papers.
Each paper has its paperId, url, year, and list of authors.
Each author has their authorId and name.
https://api.semanticscholar.org/graph/v1/author/1741101/papers?fields=citations.authors&offset=260
Returns with offset=260, and data is a list of the last 4 papers.
Each paper has its paperId and a list of citations.
Each citation has its paperId and a list of authors.
Each author has their authorId and name.
path Parameters
author_id
required
string
query Parameters
publicationDateOrYear	
string
Restricts results to the given range of publication dates or years (inclusive). Accepts the format <startDate>:<endDate> with each date in YYYY-MM-DD format.

Each term is optional, allowing for specific dates, fixed ranges, or open-ended ranges. In addition, prefixes are supported as a shorthand, e.g. 2020-06 matches all dates in June 2020.

Specific dates are not known for all papers, so some records returned with this filter will have a null value for publicationDate. year, however, will always be present. For records where a specific publication date is not known, they will be treated as if published on January 1st of their publication year.

Examples:

2019-03-05 on March 5th, 2019
2019-03 during March 2019
2019 during 2019
2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
1981-08-25: on or after August 25th, 1981
:2015-01 before or on January 31st, 2015
2015:2020 between January 1st, 2015 and December 31st, 2020
offset	
integer
Default: 0
Used for pagination. When returning a list of results, start with the element at this position in the list.

limit	
integer
Default: 100
The maximum number of results to return.
Must be <= 1000

fields	
string
A comma-separated list of the fields to be returned. See the contents of the data array in Response Schema below for a list of all available fields that can be returned. The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned. To fetch more references or citations per paper, reduce the number of papers in the batch with limit=.

Use a period (“.”) for subfields of citations and references.

Examples:

fields=title,fieldsOfStudy,references
fields=abstract,citations.url,citations.venue
Responses
200 List of papers with default or requested fields
400 Bad query parameters
404 Bad paper id

get
/author/{author_id}/papers
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"offset": 0,
"next": 0,
"data": [
{}
]
}
Snippet Text
Text snippet search
Return the text snippets that most closely match the query. Text snippets are excerpts of approximately 500 words, drawn from a paper's title, abstract, and body text, but excluding figure captions and the bibliography. It will return the highest ranked snippet first, as well as some basic data about the paper it was found in. Examples:

https://api.semanticscholar.org/graph/v1/snippet/search?query=The literature graph is a property graph with directed edges&limit=1
Returns a single snippet that is the highest ranked match.
Each snippet has text, snippetKind, section, annotation data, and score. As well as the following data about the paper it comes from: corpusId, title, authors, and openAccessInfo.

Limitations:
You must include a query.
If you don't set a limit, it will automatically return 10 results.
The max limit allowed is 1000.
query Parameters
fields	
string
A comma-separated list of the fields to be returned with each snippet element.

Paper info and the score are currently always returned. What you can specify using this fields param is which fields under the 'snippet' section (see the response schema) will be returned.

Examples:

fields=snippet.text: you'll get just the text field in the snippet section
fields=snippet.text,snippet.snippetKind: you'll get just the text and snippetKind fields in the snippet section
fields=snippet.annotations.sentences: you'll get just the sentence annotations in the snippet section
In general, you can use periods to identify nested fields (as in the examples above).

Not all fields in the response schema can be identified using this fields param though. E.g. you can't pick what you get within snippet.snippetOffset - you can either get the snippet offset with all the possible snippet offset fields, or you can not get it at all. You also can't provide paper or score or anything under paper, since those are always provided.

If you attempt to identify a field that's not supported, you'll get an error with the relevant field name. E.g.

Unrecognized or unsupported fields: [paper]

If you don't specify the fields param, you'll get a default set of fields in the snippet section. These are the default fields:

snippet.text
snippet.snippetKind
snippet.section
snippet.snippetOffset (including nested start and end)
snippet.annotations.refMentions (including nested start, end, and matchedPaperCorpusId for each element)
snippet.annotations.sentences (including nested start and end for each element)
paperIds	
string
Restricts results to snippets from specific papers. To specify papers, provide a comma-separated list of their IDs. You can provide up to approximately 100 IDs.

The following types of IDs are supported:

<sha> - a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
CorpusId:<id> - a Semantic Scholar numerical ID, e.g. CorpusId:215416146
DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
PMID:<id> - PubMed/Medline, e.g. PMID:19872477
PMCID:<id> - PubMed Central, e.g. PMCID:2323736
URL:<url> - URL from one of the sites listed below, e.g. URL:https://arxiv.org/abs/2106.15928v1
URLs are recognized from the following sites:

semanticscholar.org
arxiv.org
aclweb.org
acm.org
biorxiv.org
authors	
string
Restricts results to papers with authors matching the given names, formatted as a comma-separated list (...?authors=name1,name2,...). The search criteria are 'fuzzy', so matches that are close will also return results.


Example: galileo,kepler will return papers that include both an author similar to "galileo" and an author similar to "kepler" as co-authors. This query will also match fuzzy variations like 'keppler' and 'Kepler' (default max 'edit distance' is 2).

Important: Multiple author names are combined with AND logic, meaning results must include all specified authors. Adding more authors will narrow your results, not expand them. To search for papers by any of several authors (OR logic), perform separate searches for each author name. The maximum number of author filters is by default 10 and will return an HTTP code 400 (Bad Request) if more than 10 are supplied.

minCitationCount	
string
Restricts results to only include papers with the minimum number of citations.

Example: minCitationCount=200

insertedBefore	
string
Restricts results to snippets from papers inserted into the index before the provided date (excludes things inserted on the provided date).

Acceptable formats: YYYY-MM-DD, YYYY-MM, YYYY

publicationDateOrYear	
string
Restricts results to the given range of publication dates or years (inclusive). Accepts the format <startDate>:<endDate> with each date in YYYY-MM-DD format.

Each term is optional, allowing for specific dates, fixed ranges, or open-ended ranges. In addition, prefixes are supported as a shorthand, e.g. 2020-06 matches all dates in June 2020.

Specific dates are not known for all papers, so some records returned with this filter will have a null value for publicationDate. year, however, will always be present. For records where a specific publication date is not known, they will be treated as if published on January 1st of their publication year.

Examples:

2019-03-05 on March 5th, 2019
2019-03 during March 2019
2019 during 2019
2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
1981-08-25: on or after August 25th, 1981
:2015-01 before or on January 31st, 2015
2015:2020 between January 1st, 2015 and December 31st, 2020
year	
string
Restricts results to the given publication year or range of years (inclusive).

Examples:

2019 in 2019
2016-2020 as early as 2016 or as late as 2020
2010- during or after 2010
-2015 before or during 2015
venue	
string
Restricts results to papers published in the given venues, formatted as a comma-separated list.

Input could also be an ISO4 abbreviation. Examples include:

Nature
New England Journal of Medicine
Radiology
N. Engl. J. Med.
Example: Nature,Radiology will return papers from venues Nature and/or Radiology.

fieldsOfStudy	
string
Restricts results to papers in the given fields of study, formatted as a comma-separated list:

Computer Science
Medicine
Chemistry
Biology
Materials Science
Physics
Geology
Psychology
Art
History
Geography
Sociology
Business
Political Science
Economics
Philosophy
Mathematics
Engineering
Environmental Science
Agricultural and Food Sciences
Education
Law
Linguistics
Example: Physics,Mathematics will return papers with either Physics or Mathematics in their list of fields-of-study.

query
required
string
A plain-text search query string.

No special query syntax is supported.
limit	
integer
Default: 10
The maximum number of results to return.
Must be <= 1000

Responses
200 Best snippet match with default fields
400 Bad query parameters

get
/snippet/search
Response samples
200400
Content type
application/json

Copy
Expand allCollapse all
{
"data": [
{}
],
"retrievalVersion": "string"
}