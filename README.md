decimal
=====

This is an Erlang decimal arithmetic library.
The code is based on https://github.com/egobrain/decimal.git, but the API was reworked to be more consistant. Also some small optimizations were added.

Build
-----

    $ rebar3 compile

How To Use
----------

```
> D1 = decimal:from_binary(<<"10.22">>).
{1022,-2}

> D2 = decimal:from_binary(<<"11.245">>).
{11245,-3}

> decimal:to_binary(decimal:add(D1, D2)).
<<"21.465">>

> decimal:to_binary(decimal:mul(D1, D2)).
<<"114.92390">>

> decimal:to_binary(decimal:round(half_up, decimal:mul(D1, D2), 2)).
<<"114.92">>

> decimal:to_binary(decimal:divide(D1, D2, 10)).
<<"0.90884837705">>

> decimal:to_binary(decimal:divide(D1, D2, 2)).
<<"0.90">>

> decimal:to_binary(decimal:neg(D2)).
<<"-11.245">>

> decimal:to_binary(decimal:abs(decimal:from_binary(<<"-25.3">>))).
<<"25.3">>

> decimal:to_number(D1).
10.22

> decimal:to_binary(decimal:from_number(44.21)).
<<"44.21">>


> decimal:to_binary(decimal:from_binary(<<"1000.100">>)).
<<"1000.100">>
> decimal:to_binary(decimal:normalize(decimal:from_binary(<<"1000.100">>))).
<<"1000.1">>


> decimal:to_binary(decimal:from_binary(<<"-0.00000123456">>), #{pretty => true}). 
<<"-1.23456e-6">>

```

