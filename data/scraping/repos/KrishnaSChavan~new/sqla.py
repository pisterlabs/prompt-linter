from .. import openai
from .. import models,auth2
from fastapi import Response,status,HTTPException,Depends,APIRouter
from app.schemas import PostCreate,PostResponse,Post,LikePostOut
from app.database import get_db
from sqlalchemy.orm import Session
from typing import Optional,List
from sqlalchemy import func

def convert_to(dictionary):
    formatted_list = []
    for post, vote_count in dictionary:
        formatted_list.append({"post": post, "likes": vote_count})
    return formatted_list

router = APIRouter(
    prefix='/sqlalkk',
    tags=["sqlalkk"]
)


@router.get('/',response_model=List[LikePostOut]) #List is used to get list of elements in db in form of json
def test_p(db: Session = Depends(get_db),current_user:int = Depends(auth2.get_current_user),limit:int  = 3,skip:int = 0,search:Optional[str]= ""):
    print(limit)
    #x = db.query(models.Post).filter(models.Post.content.contains(search)).limit(limit).offset(skip).all() # limit to limit length and offset to skip some starting elements
    #result = db.query(models.Post,func.count(models.Vote.post_id).label("like")).join(models.Vote,models.Vote.post_id == models.Post.id,isouter=True).group_by(models.Post.id).all()
    results = db.query(models. Post, func.count(models.Vote.post_id).label("votes")).join(models. Vote, models.Vote.post_id == models.Post. id, isouter=True).group_by(models.Post.id).filter(models.Post.content.contains(search)).limit(limit).offset(skip).all()
    formatted_results = []
    for post, vote_count in results:
        formatted_results.append({"post": post, "likes": vote_count})

    return formatted_results
    

@router.get('/my',response_model= List[PostResponse])
def get_post(db:Session= Depends(get_db),current_user:int = Depends(auth2.get_current_user)):

    x = db.query(models.Post).filter(models.Post.owner_id == current_user.id).all()

    return x    

@router.post('/post',status_code=status.HTTP_201_CREATED,response_model= PostResponse)
def post_s(post:PostCreate,db: Session = Depends(get_db),current_user:int = Depends(auth2.get_current_user)):
    new_post = models.Post(owner_id = current_user.id, **post.dict())
    print(current_user)
    print(new_post)
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return  new_post

@router.get('/{id}',response_model=LikePostOut)
def post_g(id:int,db: Session = Depends(get_db)):
    #post = db.query(models.Post).filter(models.Post.id == id).first()
    posts = db.query(models. Post, func.count(models.Vote.post_id).label("votes")).join(models. Vote, models.Vote.post_id == models.Post. id, isouter=True).group_by(models.Post.id).filter(models.Post.id == id).first()
    print(posts)
    if not posts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'id {id} not found')
    pos = {"post": posts[0], "likes": posts[1]}
    return pos
    
@router.delete('/{id}',status_code=status.HTTP_204_NO_CONTENT)
def delete_post(id:int,db: Session = Depends(get_db),current_user:int = Depends(auth2.get_current_user)):
    post_q = db.query(models.Post).filter(models.Post.id == id)
    post = post_q.first()

    
    if post == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'post with {id} not found')

    elif  post.owner_id is not int(current_user.id):
        print(post.owner_id)
        print(current_user.id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'UNAUTHORIZED')
    
    
    
    post_q.delete(synchronize_session=False)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.put('/{id}',response_model=PostResponse)
def update_post(post:PostCreate,id:int,db: Session = Depends(get_db),current_user:int = Depends(auth2.get_current_user)):
    post_q = db.query(models.Post).filter(models.Post.id == id)

    pos = post_q.first()
    if pos == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f'id {id} not found')
    elif  pos.owner_id != int(current_user.id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail=f'UNAUTHORIZED')
    post_q.update(post.dict())
    db.commit()

    return post_q.first()


@router.post('/open')
def open(prompt:str):
    response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=100,
    n=1,
    stop=None
    )

    # Extract the generated text from the response
    generated_text = response.choices[0].text.strip()

    return {"generated_text": generated_text}


