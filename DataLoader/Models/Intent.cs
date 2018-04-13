using System;
using System.Collections.Generic;


namespace BotBotNLP.DataLoader.Models
{
  public class Intent
  {
      public string name { get; set; }
      public List<string> usersays { get; set; }
  }
}