Recommendations API (1.0)
Download OpenAPI specification:Download

Get Semantic Scholar's recommended papers given other papers as input. All methods will return up to LIMIT recommendations if they are available.

Paper Recommendations
Get recommended papers for lists of positive and negative example papers
query Parameters
limit	
integer
Default: 100
How many recommendations to return. Maximum 500.

fields	
string
A comma-separated list of the fields to be returned. See the contents of the recommendedPapers array in Response Schema below for a list of all available fields that can be returned.

The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Examples: http://api.semanticscholar.org/recommendations/v1/papers?fields=title,url,authors

Request Body schema: application/json
required
positivePaperIds	
Array of strings
negativePaperIds	
Array of strings
Responses
200 List of recommendations with default or requested fields
400 Bad query parameters
404 Input papers not found

post
/papers/
Request samples
Payload
Content type
application/json

Copy
Expand allCollapse all
{
"positivePaperIds": [
"649def34f8be52c8b66281af98ae884c09aef38b"
],
"negativePaperIds": [
"ArXiv:1805.02262"
]
}
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"recommendedPapers": [
{}
]
}
Get recommended papers for a single positive example paper
path Parameters
paper_id
required
string
query Parameters
from	
string
Default: "recent"
Enum: "recent" "all-cs"
Which pool of papers to recommend from.

limit	
integer
Default: 100
How many recommendations to return. Maximum 500.

fields	
string
A comma-separated list of the fields to be returned. See the contents of the recommendedPapers array in Response Schema below for a list of all available fields that can be returned.

The paperId field is always returned. If the fields parameter is omitted, only the paperId and title will be returned.

Examples: http://api.semanticscholar.org/recommendations/v1/papers?fields=title,url,authors

Responses
200 List of recommendations with default or requested fields
400 Bad query parameters
404 Input papers not found

get
/papers/forpaper/{paper_id}
Response samples
200400404
Content type
application/json

Copy
Expand allCollapse all
{
"recommendedPapers": [
{}
]
}