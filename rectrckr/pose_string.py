def PoseString(v):
    fields = v.split(":")
    try:
        assert len(fields) == 7
        return [float(f) for f in fields]
    except:
        raise argparse.ArgumentTypeError("7 floating-point values are needed, x:y:z:q1:q2:q3:q4")
