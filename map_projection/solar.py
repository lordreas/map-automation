"""NOAA fractional-year solar approximation; UTC in, ENU unit vector out.

One sun direction at a configurable reference site; no atmospheric refraction.
https://gml.noaa.gov/grad/solcalc/solareqns.PDF
"""
from datetime import datetime, timezone
import calendar
import math
import re
import numpy as np


def positions_for_day(day,seconds,latitude,longitude):
    """Vector form of position(), for seconds within a UTC date including 24:00.

    Uses the same astronomical approximation, without parsing a datetime per
    bisection candidate. No temporal interpolation or coarser sampling.
    """
    from datetime import timedelta
    start=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    t=np.asarray(seconds,np.float64)
    if np.any((t<0)|(t>86400)):raise ValueError('Seconds outside UTC day')
    nxt=start+timedelta(days=1);end=t==86400;hour=np.where(end,0,t/3600)
    ordinal=np.where(end,nxt.timetuple().tm_yday,start.timetuple().tm_yday)
    days=np.where(end,366 if calendar.isleap(nxt.year) else 365,366 if calendar.isleap(start.year) else 365)
    gamma=2*np.pi/days*(ordinal-1+(hour-12)/24)
    equation=229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)-.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    declination=(.006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)-.006758*np.cos(2*gamma)+.000907*np.sin(2*gamma)-.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma))
    angle=np.radians(((hour*60+equation+4*longitude)/4)%360-180);lat=math.radians(latitude)
    east=-np.cos(declination)*np.sin(angle)
    north=math.cos(lat)*np.sin(declination)-math.sin(lat)*np.cos(declination)*np.cos(angle)
    up=math.sin(lat)*np.sin(declination)+math.cos(lat)*np.cos(declination)*np.cos(angle)
    return np.degrees(np.arctan2(east,north))%360,np.degrees(np.arcsin(np.clip(up,-1,1)))


def position(iso_time, latitude, longitude):
    text=iso_time.replace("Z", "+00:00")
    text=re.sub(r"\.([0-9]{1,5})(?=[+-][0-9]{2}:[0-9]{2}$)",lambda m:"."+m.group(1).ljust(6,"0"),text)
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError("Sun time must include a UTC offset, e.g. 2026-06-21T12:00:00+02:00.")
    utc = moment.astimezone(timezone.utc)
    hour = utc.hour + utc.minute/60 + (utc.second+utc.microsecond/1e6)/3600
    gamma = 2*math.pi/(366 if calendar.isleap(utc.year) else 365)*(utc.timetuple().tm_yday-1+(hour-12)/24)
    equation = 229.18*(.000075+.001868*math.cos(gamma)-.032077*math.sin(gamma)-.014615*math.cos(2*gamma)-.040849*math.sin(2*gamma))
    declination = (.006918-.399912*math.cos(gamma)+.070257*math.sin(gamma)-.006758*math.cos(2*gamma)
                   +.000907*math.sin(2*gamma)-.002697*math.cos(3*gamma)+.00148*math.sin(3*gamma))
    angle = math.radians(((hour*60+equation+4*longitude)/4)%360-180)
    lat = math.radians(latitude)
    east = -math.cos(declination)*math.sin(angle)
    north = math.cos(lat)*math.sin(declination)-math.sin(lat)*math.cos(declination)*math.cos(angle)
    up = math.sin(lat)*math.sin(declination)+math.cos(lat)*math.cos(declination)*math.cos(angle)
    sun = np.array([east,north,up], dtype=np.float64)
    return sun, {"azimuth": math.degrees(math.atan2(east,north))%360, "altitude": math.degrees(math.asin(np.clip(up,-1,1)))}
