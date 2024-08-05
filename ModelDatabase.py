import sqlite3
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
engine = None


def connect(dbName):  #connect SQLite
    global engine
    engine = create_engine("sqlite:///" + dbName + '?check_same_thread=False')

def create():  #create table
    global engine, Base
    Base.metadata.create_all(engine)

def getSession():
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def check_repeat(mname, muploader, mdevice):
    session = getSession()

    result = session.query(Model).filter(Model.mname == mname, Model.muploader == muploader, Model.mdevice == mdevice).all()

    if len(result) == 0:
        return 0
    else:
        return result[0].mid

def upload(mname, muploader, mdevice, mscore):
    session = getSession()
    m = Model(mname = mname, muploader = muploader, mdevice = mdevice, mscore = mscore)

    try:
        session.add(m)
        session.commit()

        print('Upload Complete.')
        return True
    except:
        session.rollback()
        print('Upload Failed.')
        return False

def scoreUpdate(mid, mname, muploader, mdevice, mscore):
    session = getSession()

    try:

        session.query(Model).filter_by(mid = mid).update({'mscore': mscore})
        session.commit()

        print('Update Complete.')
        return True
    
    except:
        session.rollback()
        print('Update Failed.')
        return False
    

def getModelList():
    session = getSession()
    
    query = session.query(Model)
    
    return query.all()

def getLeaderboard(devicename):
    session = getSession()

    leaderboard = session.query(Model).filter(Model.mdevice == devicename).order_by(asc(Model.mscore))

    return leaderboard.all()


class Model(Base):
    __tablename__ = 'models'
    __table_args__ = {'sqlite_autoincrement': True}

    mid = Column(Integer, primary_key=True)
    mname = Column(String(255), nullable=False)
    muploader = Column(String(255), nullable=False)
    mdevice = Column(String(255), nullable=False)
    mscore = Column(Float)

    def __init__(self, mid=None, mname=None, muploader=None, mdevice=None, mscore=None):
        self.mid = mid
        self.mname = mname
        self.muploader = muploader
        self.mdevice = mdevice
        self.mscore = mscore