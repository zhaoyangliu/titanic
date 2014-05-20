Note: Reports need to be in pdf format and turned in via Blackboard.

### Project proposals
Project proposals are due lecture 7 at 11:59PM, and should be about 1 page in length (single spaced). It should contain title, team member names, project goals and motivation, and related work. 



### Preliminary Reports
Preliminary Reports are due lecture 15 at 11:59 PM. Each group must submit a two-page summary of their project progress by discussing 1) Steps you have completed, and any results you have obtained so far 2) The key remaining steps you plan to complete before the end of the quarter 3) Any questions or concerns you have regarding the project.


### Final Reports
Final Reports are due June 10th at 11:59PM. The final report should be about 4-5 pages in length (single spaced) and should include your project goals and motivation, along with a concise and clear statement of what results you obtained. Also mention which aspects of future work would be most interesting. Clarity in your report and presentation will contribute significantly to your grade. 


### Preliminary Reports
Information is presented as a training set regarding the survivla rate and background of passengers from the Titanic. Given the taining set information, the goal remains to predict each passenger's survival outcome from a test set of passengers.

Steps completed:
We have walked through the tutorials with Excel and Python. In excel, we generated pivot tables according to different sets of variables selected, then we located the most influential factors by observing these tables. Once a potential single-variable pattern was found, we made decisions by determining if the involved columns satisify teh pattern, with IF statement within each cell. We also tried to improve the model by extending the earlier hypothesis that maybe multiple variables have an effect o nthe results of survival. We tested age, sex, class, payments. The model we made so far is that all males will not survive, and so will the women in third class who paid more than 20 dollars for their tickets.

We also have done a same procedure with Python. We found that although Excel is good for initial submission, more complicated models will increase the inefficiency and time to manually find the proportion of survivors concerning a specific variable. Python packages offer a convenient way to search and locate different properties. We first read the data file to an object as an array, then parse that array for analysis, finally recognize the relevant pattern and apply that pattern on test.cvs.

Results:
In the film, they tried to rescue women and children first, so a good guess would be on gender. We ruled out the age factor, as the survival rate doesn't change much with or without the involvement of age.But when we consider the ticket fare, the outcome in the 3rd class starts to stratify.