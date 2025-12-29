from langchain_google_genai import ChatGoogleGenerativeAI

from  typing import TypedDict

from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model='gemini-2.0-flash')


class reviwe(TypedDict):
    summary:str
    sentiment: str


structured_ouput=model.with_structured_output(reviwe)

result=structured_ouput.invoke("A spy action movie with a runtime of 3 hrs 34 minutes approx, it's engrossing without a single moment of boredom, thanks to the robust screenplay and casting of talented performers. Director Aditya Dhar has been successful in ensuring that each character leaves an indelible mark on the viewersIt s not solely a Ranveer Singh show, as the director has ensured each character has a memorable presence. R Madhavan, Arjun Rampal, Akshay Khanna, Sanjay Dutt, and Ranveer Singh deliver performances akin to a north Indian person savoring an authentic south Indian meal, with each bite providing satisfaction. Akshay Khanna, who won hearts with Aurangzeb in Chaava, has again impressed as Rehman Dakait in Dhurandhar. His swag, look, and dialogue delivery are mind-blowing. As the movie progresses, Sanjay Dutt brings a new gear, accelerating the pace with his enjoyable screen presence. R Madhavan and Arjun Rampal have limited screen time but make their presence felt. Sara Arjun delivers a decent performance as the lady love. Ranveer Singh's character development is amazing to watch, and his mass performance will be eagerly anticipated in the second partFrom a technical standpoint, Aditya Dhars direction is astounding, perfectly capturing the essence of Dhurandhar. Sashwat Sachdevs score and music blend seamlessly with the movies toThe placement of old Bollywood classics in certain situations is entertaining and thrilling. Vikash Nowlakhas cinematography is commendable, and Shivkumar V Panickers editing is sharp")

print(result)