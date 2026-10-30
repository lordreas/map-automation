"""A monotonic playback clock, independent of render and HDMI frame rates."""
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo


def day_time(day,minutes,zone):
    naive=datetime.combine(day,datetime.min.time())+timedelta(minutes=minutes)
    value=naive.replace(tzinfo=zone)
    if value.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)!=naive:
        raise ValueError('Playback boundary falls in a missing daylight-saving time. Choose another time.')
    return value


class PlaybackClock:
    """Continuous sun motion, looping over the selected local calendar day."""
    def __init__(self,sun,now):
        self.zone=ZoneInfo(sun['timezone'])
        moment=datetime.fromisoformat(sun['time']).astimezone(self.zone)
        self.begin=day_time(moment.date(),0,self.zone).timestamp()
        self.end=day_time(moment.date(),1440,self.zone).timestamp()
        self.anchor=moment.timestamp()
        self.started=now;self.speed=sun['minutes_per_second']*60

    def sample(self,now):
        timestamp=self.begin+(self.anchor-self.begin+max(0,now-self.started)*self.speed)%(self.end-self.begin)
        return datetime.fromtimestamp(timestamp,self.zone),False


class TimeTransition:
    """A one-second smoothstep from the displayed sun to the current time."""
    def __init__(self,begin,zone,target=None,duration=1.):
        self.begin=begin.timestamp();self.zone=ZoneInfo(zone)
        self.duration=duration;self.started=None
        self.fixed_target=target.timestamp() if target is not None else None
        self.target=None

    def start(self,now,wall=None):
        if self.started is None:
            self.started=now
            self.target=self.fixed_target if self.fixed_target is not None else (wall or datetime.now(self.zone)).timestamp()+self.duration

    def sample(self,now):
        if self.started is None:return datetime.fromtimestamp(self.begin,self.zone),False
        t=min(1.,max(0.,(now-self.started)/self.duration));ease=t*t*(3-2*t)
        return datetime.fromtimestamp(self.begin+(self.target-self.begin)*ease,self.zone),t>=1
